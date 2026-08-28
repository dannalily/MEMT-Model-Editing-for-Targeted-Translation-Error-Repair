import fasttext
import argparse
import os
from utils import read_json_file, LANGS, write_json_file


class LanguageChecker:
    """FastText 语言检测器"""
    
    def __init__(self, model_path='lid.176.bin'):
        self.model = fasttext.load_model(model_path)
    
    def check(self, text, expected_lang):
        """检查文本是否为指定语言，返回 0 或 1"""
        if not text or not text.strip():
            return 0
        
        try:
            prediction = self.model.predict(text.replace('\n', ' '))
            detected = prediction[0][0].replace('__label__', '')
            
            # 统一中文代码
            if detected in ['zh-cn', 'zh-tw', 'wuu', 'yue']:
                detected = 'zh'
            
            return 1 if detected == expected_lang else 0
        except:
            return 0


def check_language_scores(data, checker, tlangs_ans):
    """检查多语言得分"""
    score_dict = {}
    for l in tlangs_ans:
        score_dict[l] = []
        for sent in data[LANGS[l]]:
            score_dict[l].append(checker.check(sent, l))
    return score_dict


def process_single_edit(checker, editing_slang, editing_tlang, translator, editings, layers, tlangs, test_type):
    """处理单次编辑"""
    print(f"Processing single edit for {editing_slang} -> {editing_tlang}")
    
    for i, edit in enumerate(editings):
        for type in test_type:
            input_file = f'editing_results/{translator}/{edit}/{editing_slang}_{editing_tlang}_{layers[editing_slang][i]}_{type}.json'
            
            if not os.path.exists(input_file):
                print(f"Skipping: {input_file} (not found)")
                continue
            
            data = read_json_file(input_file)
            tlangs_ans = [ans for ans in tlangs if ans != editing_slang]
            score_dict = check_language_scores(data, checker, tlangs_ans)
            
            save_file = f'language_check_results/single_edit/{translator}_{edit}/{editing_slang}_{editing_tlang}/{type}.json'
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            write_json_file(score_dict, save_file)
            print(f"Saved: {save_file}")


def process_batch_edit(checker, editing_slang, editing_tlang, translator, editings, layers, tlangs, test_type, seeds, batchs):
    """处理批量编辑"""
    print(f"Processing batch edit for {editing_slang} -> {editing_tlang}")
    
    for i, edit in enumerate(editings):
        for type in test_type:
            for seed in seeds:
                for batch in batchs:
                    batch_dir = f'editing_batch_results/{translator}/{edit}/batch{batch}/'
                    if not os.path.exists(batch_dir):
                        continue
                    
                    input_file = f'{batch_dir}{editing_slang}_{editing_tlang}_{layers[editing_slang][i]}_{type}-seed{seed}.json'
                    
                    if not os.path.exists(input_file):
                        print(f"Skipping: {input_file} (not found)")
                        continue
                    
                    data = read_json_file(input_file)
                    tlangs_ans = [ans for ans in tlangs if ans != editing_slang]
                    score_dict = check_language_scores(data, checker, tlangs_ans)
                    
                    save_file = f'language_check_results/batch_edit/{translator}_{edit}/batch{batch}/{editing_slang}_{editing_tlang}/{type}-seed{seed}.json'
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    write_json_file(score_dict, save_file)
                    print(f"Saved: {save_file}")


def process_sequential_edit(checker, editing_slang, editing_tlang, translator, editings, layers, tlangs, test_type, seeds, steps):
    """处理顺序编辑"""
    print(f"Processing sequential edit for {editing_slang} -> {editing_tlang}")
    
    for i, edit in enumerate(editings):
        for type in test_type:
            for seed in seeds:
                for step in steps:
                    input_file = f'editing_sequential_results/{translator}/{edit}/step{step}/{editing_slang}_{editing_tlang}_{layers[editing_slang][i]}_{type}-seed{seed}.json'
                    
                    if not os.path.exists(input_file):
                        print(f"Skipping: {input_file} (not found)")
                        continue
                    
                    data = read_json_file(input_file)
                    tlangs_ans = [ans for ans in tlangs if ans != editing_slang]
                    score_dict = check_language_scores(data, checker, tlangs_ans)
                    
                    save_file = f'language_check_results/sequential_edit/{translator}_{edit}/step{step}/{editing_slang}_{editing_tlang}/{type}-seed{seed}.json'
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    write_json_file(score_dict, save_file)
                    print(f"Saved: {save_file}")


def main():
    parser = argparse.ArgumentParser(description='Language checking for model editing results')
    parser.add_argument('--editing_type', type=str, required=True, 
                        choices=['single', 'batch', 'sequential'],
                        help='Type of editing: single, batch, or sequential')
    parser.add_argument('--editing_slang', type=str, required=True,
                        choices=['en', 'zh'],
                        help='Source language for editing')
    parser.add_argument('--editing_tlang', type=str, required=True,
                        choices=['en', 'zh', 'de', 'fr', 'ar', 'ja'],
                        help='Target language for editing')
    parser.add_argument('--model_path', type=str, default='lid.176.bin',
                        help='Path to FastText language model')
    parser.add_argument('--translator', type=str, default='Qwen2-5-3B-Instruct',
                        help='Translator model name')
    
    args = parser.parse_args()
    
    # 配置参数
    tlangs = ['en', 'zh', 'de', 'fr', 'ar', 'ja']
    editings = ['FT', 'ROME', 'MEMIT', 'AlphaEdit', 'UNKE', 'WISE', 'GRACE']
    layers = {
        'zh': ['layer5', 'layer5', 'layer2_3_4_5_6', 'layer2_3_4_5_6', 'layer5', 'layer30', 'layer30'],
        'en': ['layer7', 'layer7', 'layer5_6_7_8_9', 'layer5_6_7_8_9', 'layer7', 'layer30', 'layer30']
    }
    test_type = ['r', 'g', 'l', 'f']
    seeds = [42, 0, 1, 1234, 10, 123, 2, 5]
    batchs = [1, 4, 16, 32, 64, 128, 256]
    steps = [1, 4, 16, 64, 256, 1024]
    
    # 初始化语言检测器
    checker = LanguageChecker(args.model_path)
    
    # 根据编辑类型执行相应的处理
    if args.editing_type == 'single':
        process_single_edit(
            checker, args.editing_slang, args.editing_tlang,
            args.translator, editings, layers, tlangs, test_type
        )
    elif args.editing_type == 'batch':
        process_batch_edit(
            checker, args.editing_slang, args.editing_tlang,
            args.translator, editings, layers, tlangs, test_type, seeds, batchs
        )
    elif args.editing_type == 'sequential':
        process_sequential_edit(
            checker, args.editing_slang, args.editing_tlang,
            args.translator, editings, layers, tlangs, test_type, seeds, steps
        )
    
    print(f"\nCompleted {args.editing_type} language-check for {args.editing_slang} -> {args.editing_tlang}")


if __name__ == '__main__':
    main()