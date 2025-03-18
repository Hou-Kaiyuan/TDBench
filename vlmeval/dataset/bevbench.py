from .image_mcq import ImageMCQDataset
import pandas as pd
import os
from ..smp import load

class BEVDataset(ImageMCQDataset):
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
        return ['bevbench', 'bevbench_zoom_in']

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        # Load the TSV file from LMUData directory
        data_dir = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
        if dataset == 'bevbench':
            tsv_path = os.path.join(data_dir, 'bevbench.tsv')
        elif dataset == 'bevbench_zoom_in':
            tsv_path = os.path.join(data_dir, 'case_study_zoom_in.tsv')
        assert os.path.exists(tsv_path), f'TSV file not found at {tsv_path}.'
        return load(tsv_path)

    @staticmethod
    def dataset_type():
        return "MCQ"