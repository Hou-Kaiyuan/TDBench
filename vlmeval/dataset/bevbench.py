from .image_mcq import ImageMCQDataset
import pandas as pd
import os
from ..smp import load
from ..utils import DATASET_TYPE

class ICSLDataset(ImageMCQDataset):
    # Since we're using a local TSV, we don't need URL or MD5
    DATASET_URL = None
    DATASET_MD5 = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the TSV file from LMUData directory
        data_dir = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
        tsv_path = os.path.join(data_dir, 'icsl.tsv')
        assert os.path.exists(tsv_path), f'TSV file not found at {tsv_path}. Please run prepare_dataset.py first.'
        self.data = load(tsv_path)
    
    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate model predictions against ground truth.
        Uses the standard MCQ evaluation from parent class.
        """
        return super().evaluate(eval_file, **judge_kwargs)
    
    def build_prompt(self, line, dataset=None):
        """Build prompt for the model with image and question.
        Uses the standard MCQ prompt building from parent class.
        """
        return super().build_prompt(line, dataset)

    @staticmethod
    def dataset_type():
        return "MCQ"