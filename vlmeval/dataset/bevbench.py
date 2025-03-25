from .image_mcq import ImageMCQDataset
import pandas as pd
import os
from ..smp import load, download_file, LMUDataRoot

class BEVDataset(ImageMCQDataset):
    dataset_filenames = {
        'bevbench': 'bevbench.tsv',
        'bevbench_zoom_in': 'case_study_zoom_in.tsv',
        'bevbench_rotation': 'case_study_rotation.tsv',
        'bevbench_depth': 'bevbench_depth.tsv',
        'bevbench_temp_cs_integrity': 'temp_cs_integrity.tsv',
        'bevbench_object_counting': 'bevbench_object_count_v1.tsv',
        'bevbench_temp_cs_height': 'temp_cs_height.tsv'
    }

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate model predictions against ground truth.
        Uses the standard MCQ evaluation from parent class.
        """
        return super().evaluate(eval_file, **judge_kwargs)
    
    def build_prompt(self, line):
        """Build prompt for the model with image and question.
        Uses the standard MCQ prompt building from parent class.
        """
        msg = super().build_prompt(line)
        return msg


    @classmethod
    def supported_datasets(cls):
        return cls.dataset_filenames.keys()

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        filename = self.dataset_filenames[dataset]
        data_dir = LMUDataRoot()
        os.makedirs(data_dir, exist_ok=True)
        url = f'http://l.icsl.cc:8500/static/img/dataset/{filename}'
        tsv_path = os.path.join(data_dir, filename)
        if not os.path.exists(tsv_path):
            download_file(url, filename=tsv_path)
        assert os.path.exists(tsv_path), f'TSV file not found at {tsv_path}.'
        return load(tsv_path)

    @staticmethod
    def dataset_type():
        return "MCQ"