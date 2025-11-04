#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全离线的三元组抽取系统
使用 jieba + 规则方法，不依赖任何在线模型
"""

import jieba
import jieba.posseg as pseg
import re
from collections import defaultdict
import os

class OfflineTripleExtractor:
    """
    完全离线的三元组抽取器
    """
    
    def __init__(self):
        print("初始化离线三元组抽取系统...")
        
        # 配置 jieba
        self.init_jieba()
        
        # 初始化规则
        self.init_rules()
        
        print("✓ 离线系统初始化完成")
    
    def init_jieba(self):
        """初始化 jieba 分词器"""
        # 军事航天领域词典
        military_terms = [
            'MH-60R', '海鹰直升机', 'F/A-18F', '超级大黄蜂', '尼米兹号',
            '航空母舰', '舰载战斗机', '长征三号乙', '运载火箭', '高分十四号02星',
            '太阳同步轨道', '西昌卫星发射中心', '航天科技集团', '北斗导航系统',
            '美国海军', '太平洋舰队', '环球时报', '张军社', '宋忠平', '军事专家',
            '波音F/A-18F', '弹射逃生', '搜救队', '机组人员', '例行操作',
            '例行任务', '对地观测', '遥感应用', '高分辨率'
        ]
        
        for term in military_terms:
            jieba.add_word(term, tag='nz')
    
    def init_rules(self):
        """初始化抽取规则"""
        # 核心动词
        self.core_verbs = {
            '坠毁', '发射', '研制', '部署', '表示', '分析', '指出', '创立',
            '涉及', '进行', '执行', '成功', '取得', '推动', '救起', '逃生',
            '采用', '送入', '构建', '提供', '标志', '接受', '维持', '炫耀',
            '肩负', '执行', '导致', '震惊', '延续', '完善', '计划', '构建',
            '协同', '支撑', '迈上'
        }
        
        # 介词
        self.prepositions = {
            '在', '于', '到', '从', '向', '为', '跟', '和', '与', '据',
            '根据', '按照', '通过', '沿着'
        }
        
        # 关系动词
        self.relation_verbs = {
            '是', '为', '成为', '属于', '包括', '包含'
        }
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 清洗文本
        cleaned_text = re.sub(r'[^\u4e00-\u9fa5，。！？；：""《》\s\w]', '', text)
        
        # 分句
        sentences = re.split(r'[。！？；]', cleaned_text)
        
        # 过滤有效句子
        valid_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 4:  # 至少4个字符
                valid_sentences.append(sentence)
        
        return valid_sentences
    
    def analyze_sentence(self, sentence):
        """分析句子结构"""
        # 分词和词性标注
        words = list(pseg.cut(sentence))
        
        # 构建词语列表
        word_list = [word for word, pos in words]
        pos_list = [pos for word, pos in words]
        
        return words, word_list, pos_list
    
    def extract_triples(self, text):
        """从文本抽取三元组"""
        sentences = self.preprocess_text(text)
        all_triples = []
        
        print(f"处理 {len(sentences)} 个句子...")
        
        for i, sentence in enumerate(sentences, 1):
            print(f"\n【句子 {i}】: {sentence}")
            triples = self.extract_from_sentence(sentence)
            all_triples.extend(triples)
        
        # 后处理
        final_triples = self.post_process(all_triples)
        
        return final_triples
    
    def extract_from_sentence(self, sentence):
        """从单句抽取三元组"""
        words, word_list, pos_list = self.analyze_sentence(sentence)
        
        print(f"分词: {[(w, p) for w, p in words]}")
        
        triples = []
        
        # 应用各种规则
        triples.extend(self.rule_svo(words, word_list, pos_list, sentence))
        triples.extend(self.rule_preposition(words, word_list, pos_list, sentence))
        triples.extend(self.rule_apposition(words, word_list, pos_list, sentence))
        triples.extend(self.rule_attribution(words, word_list, pos_list, sentence))
        
        return triples
    
    def rule_svo(self, words, word_list, pos_list, sentence):
        """主谓宾规则"""
        triples = []
        
        for i, (word, pos) in enumerate(words):
            if word in self.core_verbs and pos.startswith('v'):
                # 找到动词
                verb = word
                
                # 向前找主语
                subject = self.find_entity_before(words, i)
                
                # 向后找宾语
                obj = self.find_entity_after(words, i)
                
                if subject and obj:
                    triple = self.create_triple(subject, verb, obj, 'SVO', 0.8)
                    triples.append(triple)
                    print(f"  ✓ SVO: ({subject}, {verb}, {obj})")
        
        return triples
    
    def rule_preposition(self, words, word_list, pos_list, sentence):
        """介词结构规则"""
        triples = []
        
        for i, (word, pos) in enumerate(words):
            if word in self.prepositions and pos == 'p':
                prep = word
                
                # 向前找动词
                verb = self.find_verb_before(words, i)
                
                # 向后找宾语
                obj = self.find_entity_after(words, i)
                
                if verb and obj:
                    # 找动词的主语
                    verb_index = self.find_word_index(word_list, verb)
                    if verb_index != -1:
                        subject = self.find_entity_before(words, verb_index)
                        
                        if subject:
                            relation = f"{verb}{prep}"
                            triple = self.create_triple(subject, relation, obj, 'PREP', 0.7)
                            triples.append(triple)
                            print(f"  ✓ 介词: ({subject}, {relation}, {obj})")
        
        return triples
    
    def rule_apposition(self, words, word_list, pos_list, sentence):
        """同位语规则"""
        triples = []
        
        for i, (word, pos) in enumerate(words):
            if word in self.relation_verbs and pos == 'v':
                # 向前找主语
                subject = self.find_entity_before(words, i)
                
                # 向后找宾语
                obj = self.find_entity_after(words, i)
                
                if subject and obj:
                    triple = self.create_triple(subject, '是', obj, 'APPOS', 0.9)
                    triples.append(triple)
                    print(f"  ✓ 同位语: ({subject}, 是, {obj})")
        
        return triples
    
    def rule_attribution(self, words, word_list, pos_list, sentence):
        """属性关系规则"""
        triples = []
        
        # 查找"的"字结构
        for i, (word, pos) in enumerate(words):
            if word == '的' and pos == 'uj':
                # 向前找修饰语
                modifier = self.find_entity_before(words, i)
                
                # 向后找中心语
                head = self.find_entity_after(words, i)
                
                if modifier and head:
                    triple = self.create_triple(modifier, '的', head, 'ATT', 0.85)
                    triples.append(triple)
                    print(f"  ✓ 属性: ({modifier}, 的, {head})")
        
        return triples
    
    def find_entity_before(self, words, start_index):
        """向前找实体"""
        entity_parts = []
        
        # 从start_index-1开始向前找
        i = start_index - 1
        while i >= 0:
            word, pos = words[i]
            if self.is_entity_word(pos):
                entity_parts.insert(0, word)
            elif entity_parts:  # 已经有实体词，遇到非实体词停止
                break
            i -= 1
        
        return ''.join(entity_parts) if entity_parts else None
    
    def find_entity_after(self, words, start_index):
        """向后找实体"""
        entity_parts = []
        
        # 从start_index+1开始向后找
        i = start_index + 1
        while i < len(words):
            word, pos = words[i]
            if self.is_entity_word(pos):
                entity_parts.append(word)
            elif entity_parts:  # 已经有实体词，遇到非实体词停止
                break
            i += 1
        
        return ''.join(entity_parts) if entity_parts else None
    
    def find_verb_before(self, words, start_index):
        """向前找动词"""
        for i in range(start_index-1, -1, -1):
            word, pos = words[i]
            if pos.startswith('v') and word in self.core_verbs:
                return word
        return None
    
    def find_word_index(self, word_list, target_word):
        """查找词语在列表中的位置"""
        for i, word in enumerate(word_list):
            if word == target_word:
                return i
        return -1
    
    def is_entity_word(self, pos_tag):
        """判断是否为实体词性"""
        return pos_tag.startswith(('n', 'nr', 'ns', 'nt', 'nz'))
    
    def create_triple(self, subject, relation, obj, rule, confidence):
        """创建三元组"""
        return {
            'subject': subject,
            'relation': relation,
            'object': obj,
            'subject_type': self.classify_entity(subject),
            'object_type': self.classify_entity(obj),
            'confidence': confidence,
            'rule': rule
        }
    
    def classify_entity(self, entity):
        """实体类型分类"""
        if not entity:
            return 'UNKNOWN'
        
        # 基于关键词的简单分类
        military_keywords = ['军机', '直升机', '战斗机', '航母', '火箭', '卫星', '舰队']
        location_keywords = ['南海', '西昌', '亚太', '中东', '加州', '全球']
        person_keywords = ['张军社', '宋忠平', '专家', '人员']
        org_keywords = ['海军', '航天', '集团', '中心', '时报']
        
        if any(kw in entity for kw in military_keywords):
            return 'MILITARY'
        elif any(kw in entity for kw in location_keywords):
            return 'LOCATION'
        elif any(kw in entity for kw in person_keywords):
            return 'PERSON'
        elif any(kw in entity for kw in org_keywords):
            return 'ORGANIZATION'
        else:
            return 'ENTITY'
    
    def post_process(self, triples):
        """后处理"""
        # 去重
        seen = set()
        unique_triples = []
        
        for triple in triples:
            key = (triple['subject'], triple['relation'], triple['object'])
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        # 按置信度排序
        unique_triples.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_triples

def main():
    """主函数"""
    print("=" * 70)
    print("完全离线三元组抽取系统")
    print("=" * 70)
    
    # 创建抽取器
    extractor = OfflineTripleExtractor()
    
    # 测试文本
    test_text = """据外媒报道，美国两架海军军机26日分别坠毁在南海，无人员伤亡。
    第一起坠机事件涉及一架MH-60R海鹰直升机。
    根据美国海军太平洋舰队的声明，这架直升机在尼米兹号航空母舰进行例行操作时坠入南海。
    声明称，直升机上的三名机组人员被搜救队救起。
    半小时后，一架波音F/A-18F超级大黄蜂战斗机在尼米兹号航空母舰执行例行任务时也坠毁在南海。
    机上两名机组人员成功弹射逃生，被安全救起。
    据美国海军称，所有相关人员均安全且情况稳定。两起事故原因正在调查中。"""
    
    print("\n开始抽取三元组...")
    triples = extractor.extract_triples(test_text)
    
    print(f"\n{'='*70}")
    print(f"抽取完成！共获得 {len(triples)} 个三元组")
    print(f"{'='*70}")
    
    for i, triple in enumerate(triples, 1):
        print(f"{i}. ({triple['subject']}, {triple['relation']}, {triple['object']})")
        print(f"   类型: {triple['subject_type']} -> {triple['object_type']}")
        print(f"   规则: {triple['rule']}, 置信度: {triple['confidence']}")
        print()

if __name__ == "__main__":
    main()