"""Media resolution and numeric corruption cases runnable with only stdlib."""
import csv
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from how2sign_pipeline.metadata import inspect_dataset, read_annotations
from how2sign_pipeline.npyio import _header, atomic_bytes, read_float, write_float


class MediaResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def table(self, split='train', **overrides):
        row = dict(VIDEO_NAME='source', SENTENCE_NAME=split+'.sentence', START_REALIGNED='11',
                   END_REALIGNED='12', SENTENCE='  English, punctuation | café 雨!  ', START='10', END='14')
        row.update(overrides)
        path = self.root/f'how2sign_realigned_{split}.csv'
        with path.open('w', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), delimiter='\t')
            writer.writeheader(); writer.writerow(row)
        return row, path

    def video(self, path):
        path = self.root/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture filename only; not a decodable video')
        return path

    def inspect(self, **kwargs):
        return inspect_dataset(self.root, splits=['train'], **kwargs)

    def test_missing_ambiguous_and_side_only_sources_rejected(self):
        self.table()
        self.assertIn('missing', self.inspect()['errors'][0])
        self.video('train/rgb_side/source-rgb_side.mp4')
        self.assertIn('missing', self.inspect()['errors'][0])
        self.video('train/a/rgb_front/source-rgb_front.mp4')
        self.video('train/b/rgb_front/source-rgb_front.mp4')
        self.assertIn('ambiguous', self.inspect()['errors'][0])
        self.assertFalse(self.inspect()['samples'])

    def test_front_preferred_and_all_resources_preserved(self):
        row, _ = self.table(SIGNER_ID='person1')
        front = self.video('train/rgb_front/source-rgb_front.mp4')
        side = self.video('train/rgb_side/source-rgb_side.mp4')
        self.video('train/unknown/source.mp4')
        result = self.inspect()
        self.assertFalse(result['errors'])
        record = result['samples'][0]
        self.assertEqual(record['folder'], str(front.resolve()))
        self.assertIn(str(side.resolve()), record['available_media'])
        self.assertEqual(record['signer'], 'person1')
        self.assertEqual(record['text'], row['SENTENCE'])
        self.assertEqual(record['raw_metadata']['START_REALIGNED'], '11')
        self.assertEqual(record['original_info']['START_REALIGNED'], 11)
        self.assertEqual(record['gloss'], '')

    def test_val_maps_only_to_dev_and_source_splits_are_isolated(self):
        for split in ('train','val','test'):
            self.table(split)
            self.video(f'{split}/rgb_front/source-rgb_front.mp4')
        result = inspect_dataset(self.root)
        self.assertFalse(result['errors'])
        self.assertEqual([(s['source_split'],s['split']) for s in result['samples']],
                         [('train','train'),('val','dev'),('test','test')])
        self.assertEqual(result['split_counts'], {'train':1,'val':1,'test':1})
        for sample in result['samples']:
            self.assertIn(sample['source_split'], Path(sample['folder']).parts)

    def test_duplicate_across_splits_and_ambiguous_metadata_rejected(self):
        for split in ('train','val'):
            self.table(split, SENTENCE_NAME='duplicate')
            self.video(f'{split}/rgb_front/source-rgb_front.mp4')
        result = inspect_dataset(self.root, splits=['train','val'])
        self.assertTrue(any('Duplicate sample ID across splits' in e for e in result['errors']))
        (self.root/'other_train_realigned.csv').write_bytes((self.root/'how2sign_realigned_train.csv').read_bytes())
        self.assertIn('expected one annotation table', self.inspect()['errors'][0])

    def test_sentence_fallback_is_explicit_and_timestamps_are_relative(self):
        self.table()
        self.video('train/rgb_front/sentence_level/train.sentence-rgb_front.mp4')
        self.assertIn('missing', self.inspect()['errors'][0])
        result = self.inspect(allow_sentence_clips=True)
        self.assertFalse(result['errors'])
        record = result['samples'][0]
        self.assertEqual(record['timing_reference'], 'sentence_clip_relative')
        self.assertEqual((record['start'],record['end']), (11,12))
        self.assertEqual((record['original_info']['START_REALIGNED'],record['original_info']['END_REALIGNED']), (1,2))
        self.assertEqual(record['original_info']['SOURCE_START_REALIGNED'],11)

    def test_sentence_fallback_out_of_bounds_and_missing_origin_rejected(self):
        self.video('train/rgb_front/sentence_level/train.sentence-rgb_front.mp4')
        self.table(START_REALIGNED='9')
        self.assertIn('outside sentence clip', self.inspect(allow_sentence_clips=True)['errors'][0])
        self.table(START='')
        self.assertIn('requires original START and END', self.inspect(allow_sentence_clips=True)['errors'][0])

    def test_probe_duration_failure_has_no_automatic_clamping(self):
        self.table()
        self.video('train/rgb_front/source-rgb_front.mp4')
        with patch('how2sign_pipeline.metadata.shutil.which', return_value='ffprobe'), patch(
                'how2sign_pipeline.metadata.probe_video', return_value={'duration':11.5,'width':1280,'height':720}):
            result = self.inspect(probe=True)
        self.assertIn('exceeds video duration',result['errors'][0])
        self.assertFalse(result['samples'])

    def test_reversed_equal_empty_duplicate_and_malformed_rows_rejected(self):
        for changes in ({'START_REALIGNED':'12'}, {'START_REALIGNED':'13'}, {'SENTENCE':' '}, {'VIDEO_NAME':''}):
            _, path = self.table(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                read_annotations(path,'train')
        _, path = self.table()
        text = path.read_text(encoding='utf8')
        path.write_text(text+text.splitlines()[1]+'\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError,'duplicate sample ID'): read_annotations(path,'train')
        path.write_text(text+'extra\tfield\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError,'malformed'): read_annotations(path,'train')


class NumericFormatTests(unittest.TestCase):
    def test_corrupt_empty_nan_rank_dtype_and_trailing_data_rejected(self):
        fixtures = [b'corrupt', _header('<f4',(0,4)), _header('<f4',(2,)),
                    _header('<i4',(1,1))+struct.pack('<i',1),
                    _header('<f4',(1,1))+struct.pack('<f',float('nan')),
                    _header('<f4',(1,1))+struct.pack('<f',float('inf')),
                    _header('<f4',(1,2))+struct.pack('<f',1),
                    _header('<f4',(1,1))+struct.pack('<f',1)+b'x']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'feature.npy'
            for value in fixtures:
                path.write_bytes(value)
                with self.subTest(value=value[:40]), self.assertRaises((ValueError,EOFError,SyntaxError)):
                    read_float(path)

    def test_atomic_failure_preserves_previous_complete_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'feature.npy'
            write_float(path,2,4)
            previous = path.read_bytes()
            with patch.object(Path,'replace',side_effect=OSError('simulated interrupted replacement')):
                with self.assertRaises(OSError): atomic_bytes(path,b'incomplete new content')
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(read_float(path)['shape'],(2,4))
            write_float(path,3,4)
            self.assertFalse(path.with_name(path.name+'.tmp').exists())
            self.assertEqual(read_float(path)['shape'],(3,4))


if __name__ == '__main__': unittest.main()
