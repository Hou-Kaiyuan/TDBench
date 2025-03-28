from .image_vqa import ImageVQADataset
import pandas as pd
import os
import re
from ..smp import *

class BEVGroundingDataset(ImageVQADataset):
    dataset_filenames = {
        # Grounding
        'tdbench_grounding_rot0': 'tdbench_grounding_rot0.tsv',
        'tdbench_grounding_rot90': 'tdbench_grounding_rot90.tsv',
        'tdbench_grounding_rot180': 'tdbench_grounding_rot180.tsv',
        'tdbench_grounding_rot270': 'tdbench_grounding_rot270.tsv',
    }

    
    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'iou')
        name_str = model
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        # assert model == 'iou', 'Only --judge iou is supported for BEVBench Grounding'
        # if model == 'iou':
        #     model = None
        
        data = load(eval_file)
        data = data.sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answer']]
        indexs = [str(x) for x in data['index']]
        results = {}
        
        def extract_bbox_from_string(bbox_str):
            bbox_str = bbox_str.replace('\n', '')
            parsed = re.findall(r'(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?),\s?(\d?(?:\.\d+)?)', bbox_str)
            if len(parsed) == 1:
                return float(parsed[0][0]), float(parsed[0][1]), float(parsed[0][2]), float(parsed[0][3])
            else:
                return [None, None, None, None]
        
        def calculate_bbox_iou(pred_bbox, gt_bbox):
            pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_bbox
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bbox
            
            x_min_intersect = max(pred_x_min, gt_x_min)
            y_min_intersect = max(pred_y_min, gt_y_min)
            x_max_intersect = min(pred_x_max, gt_x_max)
            y_max_intersect = min(pred_y_max, gt_y_max)
            
            if x_max_intersect < x_min_intersect or y_max_intersect < y_min_intersect:
                return 0.0
            
            intersection_area = (x_max_intersect - x_min_intersect) * (y_max_intersect - y_min_intersect)
            
            pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
            gt_area = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min)
            
            union_area = pred_area + gt_area - intersection_area
            
            iou = intersection_area / union_area
            
            return iou
        
        iou_scores = []
        individual_results = []
        
        for idx, (pred, ans, index) in enumerate(zip(predictions, answers, indexs)):
            try:
                pred_bbox = extract_bbox_from_string(pred)
                gt_bbox = extract_bbox_from_string(ans)
                
                iou = calculate_bbox_iou(pred_bbox, gt_bbox)
                iou_scores.append(iou)
                individual_results.append({
                    'index': index,
                    'prediction': pred,
                    'ground_truth': ans,
                    'iou': iou
                })
            except Exception as e:
                print(f"Error calculating IoU for index {index}: {e}")
                iou_scores.append(0.0)
                individual_results.append({
                    'index': index,
                    'prediction': pred,
                    'ground_truth': ans,
                    'iou': 0.0,
                    'error': str(e)
                })
        
        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
        
        
        data['iou'] = iou_scores  
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.xlsx')
        data.to_excel(result_file, index=False)
        
        # Save summary scores to CSV
        scores = {
            'Average IoU': avg_iou,
            'Total Samples': len(iou_scores)
        }
        
        score_df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        score_df.to_csv(score_file, index=False)
        
        return scores


    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        obj = line['question']
        question = f'\nPlease output the coordinates of the {obj} in the image in the format [x1, y1, x2, y2]. Do not include any additional text. Respond with relative coordinates between 0 and 1, with top left corner (0, 0), top right (1, 0) and bottom right (1, 1).'
        tgt_path = self.dump_image(line)

        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=question))
        return msgs

    @classmethod
    def supported_datasets(cls):
        return cls.dataset_filenames.keys()

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    # def load_data(self, dataset):
    #     filename = self.dataset_filenames[dataset]
    #     data_dir = LMUDataRoot()
    #     os.makedirs(data_dir, exist_ok=True)
    #     url = f'http://l.icsl.cc:8500/static/img/dataset/{filename}'
    #     tsv_path = os.path.join(data_dir, filename)
    #     if not os.path.exists(tsv_path):
    #         download_file(url, filename=tsv_path)
    #     assert os.path.exists(tsv_path), f'TSV file not found at {tsv_path}.'
    #     return load(tsv_path)

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
        return "VQA"