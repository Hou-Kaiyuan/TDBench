from .image_mcq import ImageMCQDataset
import pandas as pd
import os
from ..smp import load, download_file, LMUDataRoot

class BEVDataset(ImageMCQDataset):
    dataset_filenames = {
        # TD Bench
        'tdbench_rot0': 'tdbench_rot0.tsv',
        'tdbench_rot90': 'tdbench_rot90.tsv',
        'tdbench_rot180': 'tdbench_rot180.tsv',
        'tdbench_rot270': 'tdbench_rot270.tsv',

        # Grounding
        'tdbench_grounding_rot0': 'tdbench_grounding_rot0.tsv',
        'tdbench_grounding_rot90': 'tdbench_grounding_rot90.tsv',
        'tdbench_grounding_rot180': 'tdbench_grounding_rot180.tsv',
        'tdbench_grounding_rot270': 'tdbench_grounding_rot270.tsv',

        # Case Study
        'tdbench_cs_zoom': 'case_study_zoom_in.tsv',
        'tdbench_cs_height': 'case_study_height.tsv',
        'tdbench_cs_integrity': 'case_study_integrity.tsv',
        'tdbench_cs_depth': 'case_study_depth.tsv',
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