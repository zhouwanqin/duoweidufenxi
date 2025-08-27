
import os, re, math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

def convert_to_utf8(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        for enc in ['utf-8', 'gbk', 'iso-8859-1']:
            try:
                content = raw_data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = raw_data.decode('utf-8', errors='replace')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def build_vocabulary_df(vocab_dir: str) -> pd.DataFrame:
    # Convert all txt to UTF-8 and assemble into a DataFrame
    for root, _, files in os.walk(vocab_dir):
        for file in files:
            if file.endswith(".txt"):
                convert_to_utf8(os.path.join(root, file))

    df_vocabulary = pd.DataFrame()
    for root, _, files in os.walk(vocab_dir):
        for file in files:
            if file.endswith(".txt"):
                fp = os.path.join(root, file)
                with open(fp, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    df_vocabulary[file] = pd.Series(lines)
    df_vocabulary = df_vocabulary.T
    return df_vocabulary

def read_txt_keep_pos(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def slice_text(text: str, n: int) -> List[str]:
    return [text[i:i+n] for i in range(0, len(text), n)] if n>0 else [text]

def collect_corpus_chunks(txt_paths: List[str], do_slice: bool, slice_length: int) -> List[Tuple[str,str]]:
    # Return list of (chunk_id, chunk_text).
    chunks = []
    for p in txt_paths:
        txt = read_txt_keep_pos(p)
        base = os.path.basename(p)
        if do_slice:
            parts = slice_text(txt, slice_length)
            for i, t in enumerate(parts, 1):
                chunks.append((f"{base}#part{i}", t))
        else:
            chunks.append((base, txt))
    return chunks

class VocabularyMatcher:
    def __init__(self, df_vocabulary, df_text_split):
        self.df_vocabulary = df_vocabulary
        self.df_text_split = df_text_split
        self.vocab_lists = {}
        self.regex_patterns = {}
        categories = [
            ('指示代词', '词单61_指示代词.txt'),
            ('口语用第一人称代词', '词单63_口语用第一人称代词.txt'),
            ('时间副词', '词单64_时间副词.txt'),
            ('高频副词', '词单65_高频副词.txt'),
            ('不定代词', '词单66_不定代词.txt'),
            ('心理动词', '词单28_心理动词.txt'),
            ('中频副词', '词单24_中频副词.txt'),
            ('动作行为动词', '词单16_动作行为动词.txt'),
            ('结果动词', '词单16-1_结果动词.txt'),
            ('瞬成动词', '词单16-2_瞬成动词.txt'),
            ('动作动词', '词单16-3_动作动词.txt'),
            ('静态动词', '词单16-4_静态动词.txt'),
            ('推测性动词', '词单27_推测性动词.txt'),
            ('必然性副词', '词单33_必然性副词.txt'),
            ('副词性模糊限制语', '模糊限制语.txt'),
            ('肯定性动词', '词单69_肯定性动词.txt'),
            ('可能性副词', '词单35_可能性副词.txt'),
            ('交际动词', '词单26_交际动词.txt'),
            ('第三人称代词', '词单25_第三人称代词.txt'),
            ('中频名词', '词单1_中频名词.txt'),
            ('抽象名词', '词单2_抽象名词.txt'),
            ('命题性名词', '词单4_命题性名词.txt'),
            ('大学学科专业分类词', '词单7_大学学科专业分类词.txt'),
            ('心理名词', '词单9-4_心理名词.txt'),
            ('使役动词', '词单10_使役动词.txt'),
            ('中频形容词', '词单11_中频形容词.txt'),
            ('中频动词', '词单12_中频动词.txt'),
            ('低频形容词', '词单17_低频形容词.txt'),
            ('低频名词', '词单18_低频名词.txt'),
            ('低频副词', '词单19_低频副词.txt'),
            ('具象名词', '词单29_具象名词.txt'),
            ('具象科技名词', '词单30_具象科技名词.txt'),
            ('度量衡名词', '词单31_度量衡名词.txt'),
            ('态度性副词', '词单34_ 态度性副词.txt'),
            ('高频名词', '词单72_高频名词.txt'),
            ('高频动词', '词单73_高频动词.txt'),
            ('高频形容词', '词单74_高频形容词.txt'),
            ('口语词', '口语词.txt'),
            ('插入语', '插入语.txt'),
            ('儿化音', '儿化音.txt'),
            ('嵌偶单音词', '词单42_嵌偶单音词.txt'),
            ('合偶双音词', '词单43_合偶双音词.txt'),
            ('古语词', '古语词.txt'),
            ('存现动词', '词单20_存现动词.txt'),
            ('样态形容词','词单21_样态形容词.txt'),
            ('低频动词','词单22_低频动词.txt'),
            ('第二人称代词','词单13_第二人称代词.txt')
        ]
        for category, index in categories:
            vocab_list = self._get_vocab_list(index)
            self.vocab_lists[category] = vocab_list
            if vocab_list:
                self.regex_patterns[category] = re.compile('|'.join(map(re.escape, vocab_list)))
            else:
                self.regex_patterns[category] = re.compile('$^')  # match nothing

    def _get_vocab_list(self, index):
        d1 = self.df_vocabulary.loc[index].tolist() if index in self.df_vocabulary.index else []
        return [str(item) for item in d1 if str(item).strip()]

    def _match_single_row(self, index):
        text = str(index)
        row_results = {}
        for category, pattern in self.regex_patterns.items():
            matches = pattern.findall(text)
            row_results[category] = len(matches)
        return row_results

    def match_all_vocabularies(self):
        with ThreadPoolExecutor() as executor:
            all_results = list(executor.map(self._match_single_row, self.df_text_split.index))
        for category in self.vocab_lists.keys():
            self.df_text_split[category] = [result.get(category, 0) for result in all_results]
        return self.df_text_split

def count_label(df):
    df['无生性人称代词'] = [str(idx).count('它/rr') + str(idx).count('它们/rr') for idx in df.index]
    df['疑问代词'] = [str(idx).count('/ry') for idx in df.index]
    df['介词'] = [str(idx).count('/p')  for idx in df.index]
    df['对等连接词'] = [str(idx).count('/cc') for idx in df.index]
    df['进行式时貌词'] = [str(idx).count('/uzhe') for idx in df.index]
    df['过去式时貌词'] = [str(idx).count('/ule') for idx in df.index]
    df['过去完成式时貌词'] = [str(idx).count('/uguo') for idx in df.index]
    df['“的”的名词化功能词'] = [str(idx).count('/vn') + str(idx).count('/an') for idx in df.index]
    df['介词“把”'] = [str(idx).count('/pba') for idx in df.index]
    df['系词'] = [str(idx).count('是/vshi') for idx in df.index]
    df['比况助词'] = [str(idx).count('/uyy') for idx in df.index]
    df['状态形容词'] = [str(idx).count('/z') for idx in df.index]
    df['区别形容词'] = [str(idx).count('/b') for idx in df.index]
    df['领属词缀'] = [str(idx).count('/ude1') for idx in df.index]
    df['副词化功能词'] = [str(idx).count('/ude2') for idx in df.index]
    df['结果补语词'] = [str(idx).count('/ude3') for idx in df.index]
    df['模拟语助词'] = [str(idx).count('/udeng') for idx in df.index]
    df['副动词'] = [str(idx).count('/vd') for idx in df.index]
    df['趋向动词'] = [str(idx).count('/vf') for idx in df.index]
    df['形式动词'] = [str(idx).count('/vx') for idx in df.index]
    df['感叹词'] = [str(idx).count('/e') + str(idx).count('/y') for idx in df.index]
    df['数量词'] = [str(idx).count('/m') + str(idx).count('/mq') for idx in df.index]
    df['量化词'] = [str(idx).count('/q') + str(idx).count('/qv') +str(idx).count('/qt')for idx in df.index]
    df['拟声词'] = [str(idx).count('/o') for idx in df.index]
    df['第一人代词“我”'] = [str(idx).count('我/rr') for idx in df.index]
    df['人称代词复数型'] = [str(idx).count('我们/rr') + str(idx).count('咱们/rr') +str(idx).count('俺们/rr')+ str(idx).count('你们/rr') +str(idx).count('它们/rr')+ str(idx).count('他们/rr') +str(idx).count('她们/rr')for idx in df.index]
    return df

def count_Regular(df):
    for idx in df.index:
        list1 = str(idx).split(' ')
        count = sum(
            (
             ('好' in list1[i] and '好' in list1[i + 1]) or 
             ('行' in list1[i] and '行' in list1[i + 1]) or
             ('对' in list1[i] and '对' in list1[i + 1]) or
             ('是' in list1[i] and '是' in list1[i + 1]) or
             ('好' in list1[i] and '了' in list1[i + 1])
             )
            for i in range(len(list1) - 1)
        )
        df.loc[idx, '话语标志'] = count

        count = sum(
            (
            (any(word in list1[i] for word in ('正在', '正', '在')) and any(tag in list1[i + 1] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg"))) or
            (any(tag in list1[i] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")) and any(word in list1[i + 1] for word in ('着', '中', '之中'))) or
            (any(tag in list1[i] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")) and
            any(n_tag in list1[i + 1] for n_tag in ("n", 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng') if n_tag not in ("wn", "udeng", "ulian", "an", "vn")) and
            any(word in list1[i + 2] for word in ('中', '来')) and
            '着' in list1[i + 3]) or
            (any(tag in list1[i] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")) and
            any(word in list1[i + 1] for word in ('时', '之时', '的')) and
            '时候' in list1[i + 2]) or
            (any(tag in list1[i] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")) and
            '呢' in list1[i + 1]) or
            (any(tag in list1[i] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")) and
            any(n_tag in list1[i + 1] for n_tag in ("n", 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng') if n_tag not in ("wn", "udeng", "ulian", "an", "vn")) and
            '呢' in list1[i + 2]) or
            ('一边' in list1[i] and any(tag in list1[i + 1] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg", 'w')) and
            '一边' in list1[i + 2] and any(tag in list1[i + 3] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg"))) or
            ('一边' in list1[i] and any(tag in list1[i + 1] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")) and
            '一边' in list1[i + 2] and any(tag in list1[i + 3] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg"))) or
            ('边' in list1[i] and any(tag in list1[i + 1] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")) and
            '边' in list1[i + 2] and any(tag in list1[i + 3] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg"))) or
            ('边' in list1[i] and any(tag in list1[i + 1] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg", 'w')) and
            '边' in list1[i + 2] and any(tag in list1[i + 3] for tag in ("v", "vn", "vf", "vx", "vi", "vl", "vg")))
            )
            for i in range(len(list1) - 3)
        )
        df.loc[idx, '表进行的句式'] = count

        count = sum(
            (
             ('来' in list1[i] and 'm' in list1[i + 1] and 'p' in list1[i + 2]) or
             ('来' in list1[i] and '上' in list1[i + 1]) or
             ('来' in list1[i] and 'a' in list1[i + 1] and '的' in list1[i + 2]) or
             ('来' in list1[i] and 'p' in list1[i + 1]) or
             ('按' in list1[i] and '来' in list1[i + 1]) or 
             ('着' in list1[i] and '来' in list1[i + 1]) or 
             ('来' in list1[i] and '得' in list1[i + 1] and 'a' in list1[i + 2]) 
             )
            for i in range(len(list1) - 2)
        )
        df.loc[idx, '约略动词'] = count

        count = sum(
            (
            ('v' in list1[i] and 'uguo' in list1[i + 1]) or
            ('v' in list1[i] and 'ule' in list1[i + 1]) or
            (any(word in list1[i] for word in ('曾经', '已然', '曾', '业已', '业经', '刚', '刚刚', '一度', '本', '本来', '原本', '原来')) and 'v' in list1[i + 1])
             )
            for i in range(len(list1) - 1)
        )
        df.loc[idx, '表过去的句式'] = count

        count = sum(
            (
             (any(word in list1[i] for word in ('a', 'v')) and any(word in list1[i + 1] for word in ('度', '性'))) or
             (any(word in list1[i] for word in ('度/n', '性/n')))
             )
            for i in range(len(list1) - 1)
        )
        df.loc[idx, '名物化'] = count

        count = sum(
            ('n' in list1[i] or 'r' in list1[i]) and any(keyword in list1[i + 1] for keyword in ('被', '挨', '遭', '遇', '受', '获', '承', '蒙')) and 'v' in list1[i + 2]
            for i in range(len(list1) - 2)
        )
        df.loc[idx, '无施事者被动句'] = count

        count = sum(
            any(keyword in list1[i] for keyword in ('要','应','该','应该','应当','该当','一定','必须','务必')) and 'v' in list1[i + 1]
            for i in range(len(list1) - 1)
        )
        df.loc[idx, '必要性情态动词'] = count

        count = sum(
            (any(n_tag in list1[i] for n_tag in ("n", 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng') if n_tag not in ("wn", "udeng", "ulian", "an", "vn")) and
            any(word in list1[i + 1] for word in ('们')))
            for i in range(len(list1) - 1)
        )
        df.loc[idx, '普通名词复数形'] = count

        count = sum(
            (
            ('pbei' in list1[i]) or
            ('由' in list1[i] and any(n_tag in list1[i+1] for n_tag in ("n", 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng') if n_tag not in ("wn", "udeng", "ulian", "an", "vn"))) or
            ('为' in list1[i] and any(n_tag in list1[i+1] for n_tag in ("n", 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng') if n_tag not in ("wn", "udeng", "ulian", "an", "vn")) and '所' in list1[i+2] and 'v' in list1[i+3]) or
            (any(n_tag in list1[i] for n_tag in ("教", '叫', '让', '给')) and any(n_tag in list1[i+1] for n_tag in ("n", 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng') if n_tag not in ("wn", "udeng", "ulian", "an", "vn")) and 'v' in list1[i+2]) 
            )
            for i in range(len(list1) - 3)
        )
        df.loc[idx, '被动句式'] = count        
    return df

def count_Regular_vocabulary(df_vocabulary, result_df):
    phrase_types = {
        "词单75-10_转折复句.txt": "转折复句",
        "词单75-9_目的复句.txt": "目的复句",
        "词单75-8_因果复句.txt": "因果复句",
        "词单75-7_假设复句.txt": "假设复句",
        "词单75-6_条件复句.txt": "条件复句",
        "词单75-5_递进复句.txt": "递进复句",
        "词单75-4_选择复句.txt": "选择复句",
        "词单75-3_解说复句.txt": "解说复句",
        "词单75-2_顺承复句.txt": "顺承复句",
        "词单75-1_并列复句.txt": "并列复句",
    }
    compiled_patterns_dict = {}
    for file_name, column_name in phrase_types.items():
        if file_name in df_vocabulary.index:
            turning_phrases_row = df_vocabulary.loc[file_name]
            turning_phrases = [phrase for phrase in turning_phrases_row if pd.notna(phrase)]
            turning_phrases = [phrase.replace(" ", "").replace("　", "") for phrase in turning_phrases]
            combined_pattern = '|'.join(re.escape(pattern) for pattern in turning_phrases)
            if combined_pattern:
                compiled_patterns_dict[column_name] = re.compile(combined_pattern)

    for column_name, compiled_pattern in compiled_patterns_dict.items():
        def count_matches(text):
            matches = compiled_pattern.findall(text)
            return len(matches) if matches else 0
        result_df[column_name] = result_df.index.map(count_matches)
    return result_df

def count_mix(df_vocabulary, result_df):
    phrase_types = {
        "词单60_缩略语.txt": "缩略语",
        "词单68_处所副词.txt": "方位成分",
        "词单3_范畴形容词.txt": "范畴形容词",
        "词单14_可能性情态动词.txt": "可能性情态动词",
        "词单71_程度副词.txt": "程度副词",
        "词单70_否定词.txt": "否定词",
        "词单75-3_解说复句.txt": "解说复句",
        "词单36_序数词.txt": "序数词",
        "词单9_指人泛指名词2.txt": "指人泛指名词",
        "词单8_指人泛指名词1.txt": "指人泛指名词",
        "词单6_集体名词.txt": "集体名词"
    }
    for file_name, column_name in phrase_types.items():
        if file_name not in df_vocabulary.index:
            continue
        turning_phrases_row = df_vocabulary.loc[file_name]
        turning_phrases = [phrase for phrase in turning_phrases_row if pd.notna(phrase)]
        turning_phrases = [phrase.replace(' ', '').replace('　', '') for phrase in turning_phrases]
        compiled_patterns = [re.compile(re.escape(pattern)) for pattern in turning_phrases]

        for idx in result_df.index:
            count = 0
            if column_name == "缩略语":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                result_df.loc[idx, column_name] = count

            if column_name == "方位成分":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                    count += idx.count('rys')
                    count += idx.count('rzs')
                    count += idx.count('s ')    
                    count += idx.count('f ')
                result_df.loc[idx, column_name] = count

            if column_name == "范畴形容词":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                    count += idx.count('一般的') + idx.count('普通的') + idx.count('大体的') + idx.count('综合的') + idx.count('全部的') + idx.count('完全的') + idx.count('总计的') + idx.count('各种各样的') + idx.count('多方面的')
                result_df.loc[idx, column_name] = count 

            if column_name == "程度副词":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                    count += idx.count('深深') + idx.count('极力') + idx.count('更甚') + idx.count('至为') + idx.count('透顶') + idx.count('绝顶') + idx.count('大大') + idx.count('全然') + idx.count('极大')              
                result_df.loc[idx, column_name] = count

            if column_name == "否定词":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                result_df.loc[idx, column_name] = count  

            if column_name == "序数词":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                    dix_s = str(idx).split(' ')
                    count += sum('m' in dix_s[i] and ('md' in dix_s[i + 1] or 'mn' in dix_s[i + 1]) for i in range(len(dix_s) - 2))
                result_df.loc[idx, column_name] = count

            if column_name == "可能性情态动词":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                    dix_s = str(idx).split(' ')
                    count += sum('看' in dix_s[i] and '起来' in dix_s[i + 1] and '好像' in dix_s[i + 2] for i in range(len(dix_s) - 2))
                    count += sum('看' in dix_s[i] and '样子' in dix_s[i+1] for i in range(len(dix_s) - 2))
                result_df.loc[idx, column_name] = count

            if column_name == "指人泛指名词":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                result_df.loc[idx, column_name] = count

            if column_name == "集体名词":
                for pattern in compiled_patterns:
                    count += len(pattern.findall(idx))
                    count += idx.count('/nt')              
                result_df.loc[idx, column_name] = count
    return result_df

def basic(result_df):
    def clean_text(text):
        return re.sub(r'/[a-zA-Z0-9]+', '', text)
    def count_tokens(text):
        return len([word for word in text.split() if word.strip() != ''])
    def count_sentences(text):
        return len(re.findall(r'[。！？]', text))
    def count_chars(text):
        return len(re.sub(r'[^\w\s]', '', text))

    for idx in result_df.index:
        text = idx
        clean_text_content = clean_text(text)
        tokens = count_tokens(clean_text_content)
        sentences = count_sentences(text)
        chars = count_chars(clean_text_content)
        lexical_diversity = len(set(clean_text_content.split())) / math.sqrt(tokens) if tokens > 0 else 0
        avg_word_length = chars / tokens if tokens > 0 else 0
        avg_sentence_length = chars / sentences if sentences > 0 else 0
        result_df.loc[idx, '词汇多样性'] = lexical_diversity
        result_df.loc[idx, '平均词长'] = avg_word_length
        result_df.loc[idx, '平均句长'] = avg_sentence_length
    return result_df

def clean_final_text(s: str) -> str:
    s = re.sub(r'/[a-zA-Z0-9]+', '', s)
    s = s.replace(' ', '')
    return s

def build_features_for_corpus(txt_paths, df_vocabulary, feature_excel_path: str, do_slice: bool, slice_length: int):
    logs = []
    # 读取 Excel 的特征列名
    df_excel = pd.ExcelFile(feature_excel_path).parse()
    col_name = df_excel.columns[0] if len(df_excel.columns)>0 else "特征"
    column_names = df_excel[col_name][1:].dropna().tolist()

    # 收集文本分片
    chunks = collect_corpus_chunks(txt_paths, do_slice=do_slice, slice_length=slice_length)
    idx_texts = [c[1] for c in chunks]
    ids = [c[0] for c in chunks]

    # 初始化特征表
    df_text_split = pd.DataFrame(index=idx_texts, columns=column_names)

    logs.append(f"已收集文本片段：{len(chunks)}")
    logs.append("开始匹配词单特征……")
    matcher = VocabularyMatcher(df_vocabulary, df_text_split)
    result_df = matcher.match_all_vocabularies()

    logs.append("统计标签特征……")
    result_df = count_label(result_df)

    logs.append("统计正则表达式特征……")
    result_df = count_Regular(result_df)

    logs.append("统计复句/短语特征……")
    result_df = count_Regular_vocabulary(df_vocabulary, result_df)

    logs.append("统计混合特征……")
    result_df = count_mix(df_vocabulary, result_df)

    logs.append("计算基本语言学特征……")
    result_df = basic(result_df)

    logs.append("清洗索引文本……")
    result_df.index = [clean_final_text(idx) for idx in result_df.index]
    result_df = result_df.dropna(axis=1, how='all')

    # 用片段 id 替换为更清晰的行索引（防止不同片段文本清洗后相同而冲突）
    result_df.insert(0, 'ChunkID', ids)
    result_df = result_df[~result_df.index.duplicated(keep='first')]
    result_df = result_df.set_index('ChunkID')

    return result_df, logs
