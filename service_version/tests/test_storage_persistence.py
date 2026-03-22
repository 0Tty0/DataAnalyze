import os
import shutil
import unittest
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.storage import Storage


class StoragePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent / '_tmp_persist' / uuid.uuid4().hex
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workspace / 'storage.db'

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_thread_and_message_persist_after_restart(self) -> None:
        s1 = Storage(workspace_root=str(self.workspace), db_path=str(self.db_path))
        tid = s1.create_thread(summary='这是一个用于持久化测试的话题', owner_id='user-a')
        s1.append_thread_message(tid, 'user', '请分析本周销售数据')

        s2 = Storage(workspace_root=str(self.workspace), db_path=str(self.db_path))
        self.assertTrue(s2.has_thread(tid))

        row = s2.get_thread(tid)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.get('owner_id'), 'user-a')
        self.assertGreaterEqual(int(row.get('message_count', 0)), 1)

        msgs = s2.list_thread_messages(tid, limit=10)
        self.assertTrue(msgs)
        self.assertEqual(msgs[-1].get('role'), 'user')

    def test_file_metadata_persist_after_restart(self) -> None:
        s1 = Storage(workspace_root=str(self.workspace), db_path=str(self.db_path))
        f = s1.create_file('demo.csv', b'a,b\n1,2\n', 'file-extract')

        s2 = Storage(workspace_root=str(self.workspace), db_path=str(self.db_path))
        file_obj = s2.get_file(f.id)
        self.assertIsNotNone(file_obj)
        assert file_obj is not None
        self.assertEqual(file_obj.filename, 'demo.csv')
        self.assertTrue(os.path.exists(s2.get_file_path(f.id) or ''))


if __name__ == '__main__':
    unittest.main()
