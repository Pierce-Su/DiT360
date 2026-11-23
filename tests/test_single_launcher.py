import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('single', Path(__file__).parents[1] / 'scripts/run_single_i2p.py')
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)


class LauncherSafetyTests(unittest.TestCase):
    def test_requested_output(self):
        self.assertEqual(str(single.RUN), '/mnt/NAS/E3DQA_project/panoramas/dit360/first-sample-t_db92abb3')

    def test_existing_target_refused_before_gpu_query(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(single, 'RUN', Path(temp)), patch('sys.argv', ['run']), patch.object(single, 'query') as query:
            with self.assertRaisesRegex(SystemExit, 'Refusing existing output'):
                single.main()
            query.assert_not_called()

    def test_any_existing_job_requires_coordination(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / 'DOWNLOAD_DONE').touch()
            with patch.object(single, 'WORK', work), patch.object(single, 'RUN', work/'absent'), patch('sys.argv', ['run']), patch.object(single, 'query', return_value=[['123', '1']]):
                with self.assertRaisesRegex(SystemExit, 'WAITING_FOR_GPU_COORDINATION'):
                    single.main()
                self.assertFalse((work/'absent').exists())
