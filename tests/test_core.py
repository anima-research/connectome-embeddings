import asyncio
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from index_store import IndexRegistry, NamespaceIndex
from server import Worker, bucket_len, pin_text_only

class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = IndexRegistry(self.temp.name, "test-model", 3)
        self.idx = self.registry.get("test/session", create=True)
    def tearDown(self):
        for idx in self.registry.open.values(): idx.db.close()
        self.temp.cleanup()
    def put(self, items, vectors):
        pending, unchanged = self.idx.plan_upsert(items)
        self.assertFalse(unchanged)
        return self.idx.commit_vectors(pending, np.array(vectors, dtype=np.float32))
    def test_insert_search_filter_delete(self):
        self.assertEqual(self.put([
            {"id":"a","text":"alpha","channel":"one","kind":"message","ts":10},
            {"id":"b","text":"beta","channel":"two","kind":"summary","ts":20},
        ], [[1,0,0],[0,1,0]]), (2,0))
        q=np.array([1,0,0],dtype=np.float32)
        self.assertEqual(self.idx.search(q,k=1)[0]["id"],"a")
        self.assertEqual([h["id"] for h in self.idx.search(q,channel="two")],["b"])
        self.assertEqual(self.idx.delete(["a"]),1)
        self.assertEqual(self.idx.delete(["a"]),0)
        self.assertEqual(self.idx.stats()["count"],1)
        self.assertEqual([h["id"] for h in self.idx.search(q)],["b"])
    def test_unchanged_text_refreshes_metadata(self):
        self.put([{"id":"a","text":"alpha","channel":"old"}],[[1,0,0]])
        pending,unchanged=self.idx.plan_upsert([{"id":"a","text":"alpha","channel":"new","cursor":42}])
        self.assertEqual(pending,[]);self.assertEqual(unchanged,["a"])
        hit=self.idx.search(np.array([1,0,0],dtype=np.float32))[0]
        self.assertEqual(hit["channel"],"new")
        self.assertEqual(self.idx.stats()["by_kind"]["?"]["max_cursor"],42)
    def test_changed_text_reembeds_and_persists(self):
        self.put([{"id":"a","text":"alpha"}],[[1,0,0]])
        self.assertEqual(self.put([{"id":"a","text":"revised"}],[[0,1,0]]),(0,1))
        other=NamespaceIndex(self.idx.path,"test/session","test-model",3)
        try:
            h=other.search(np.array([0,1,0],dtype=np.float32))[0]
            self.assertEqual(h["text"],"revised");self.assertEqual(h["score"],1.0)
        finally: other.db.close()
    def test_missing_and_invalid_namespace(self):
        self.assertIsNone(self.registry.get("absent"))
        for ns in ["", ".hidden", "x"*201]:
            with self.assertRaises(ValueError):self.registry.get(ns,create=True)

class HelperTests(unittest.TestCase):
    def test_length_buckets_monotone_and_bounded(self):
        previous=0
        for n in range(1,8193):
            b=bucket_len(n,8192)
            self.assertGreaterEqual(b,n);self.assertLessEqual(b,8192)
            self.assertGreaterEqual(b,previous);previous=b
        self.assertEqual(bucket_len(17,8192),32)
    def test_text_pin_blocks_media_resolution(self):
        mod=types.ModuleType("_embedding_test_model")
        mod._resolve_input=lambda x:("media",x)
        mod._is_media_string=lambda x:mod._resolve_input(x)[0]!="text"
        Layer=type("Layer",(),{"__module__":mod.__name__})
        with patch.dict(sys.modules,{mod.__name__:mod}):
            pin_text_only([Layer()])
            self.assertEqual(mod._resolve_input("https://example.invalid/image.jpg"),("text","https://example.invalid/image.jpg"))
            self.assertFalse(mod._is_media_string(sys.executable))
    def test_text_pin_refuses_unknown_model_interface(self):
        with self.assertRaises(RuntimeError):pin_text_only([object()])
    def test_worker_batch_limits_preserve_input_order(self):
        class FakeModel:
            max_seq_length=128
            def __init__(self): self.calls=[]
            def tokenizer(self,texts,**kw):return {"input_ids":[[0]*len(t) for t in texts]}
            def get_sentence_embedding_dimension(self):return 3
            def encode(self,texts,**kw):
                self.calls.append(list(texts))
                return [[len(t),0,0] for t in texts]
        loop=asyncio.new_event_loop()
        try:
            model=FakeModel();w=Worker(model,loop,15,256,8,16,attn_budget=64)
            vectors,lengths=w._encode(["aa","bbbbbbbb","cccccccc"],"retrieval","document")
            self.assertEqual(lengths,[2,8,8])
            self.assertEqual(vectors[:,0].tolist(),[2,8,8])
            for batch in model.calls:self.assertLessEqual(len(batch)*max(map(len,batch))**2,64)
        finally:loop.close()

if __name__ == "__main__":unittest.main()
