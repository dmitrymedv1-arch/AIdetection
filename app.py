"""
Streamlit приложение для анализа научных статей на предмет использования ИИ
Многоуровневый анализ: от поверхностных артефактов до глубоких семантических паттернов
"""

import streamlit as st
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import hashlib
from datetime import datetime
import tempfile
import os

# NLP библиотеки
import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from scipy import stats

# Для работы с документами
from docx import Document
import subprocess
import io
import pydantic
import warnings

# Проверяем версию Pydantic и выдаем предупреждение если нужно
if pydantic.__version__.startswith('2'):
    warnings.warn(
        "Вы используете Pydantic v2. spaCy может работать некорректно. "
        "Рекомендуется установить Pydantic v1: pip install pydantic==1.10.14",
        RuntimeWarning
    )

# Конфигурация страницы
st.set_page_config(
    page_title="AI Detector for Scientific Papers",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 5px solid #1E88E5;
    }
    .warning-text {
        color: #f39c12;
        font-weight: bold;
    }
    .danger-text {
        color: #e74c3c;
        font-weight: bold;
    }
    .success-text {
        color: #27ae60;
        font-weight: bold;
    }
    .info-box {
        background-color: #e1f5fe;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Классы и функции для анализа
# ============================================================================

class UnicodeArtifactDetector:
    """Детектор Unicode-артефактов (уровень 1)"""
    
    def __init__(self):
        # Надстрочные и подстрочные символы (U+2070–U+209F)
        self.sup_sub_chars = set([
            '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',  # надстрочные цифры
            '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',  # подстрочные цифры
            '⁺', '⁻', '⁼', '⁽', '⁾',  # надстрочные знаки
            '₊', '₋', '₌', '₍', '₎',  # подстрочные знаки
            'ª', 'º',  # женский/мужской ординар
        ])
        
        # Fullwidth цифры (U+FF10–U+FF19) - выглядят как обычные, но другие коды
        self.fullwidth_digits = set([chr(i) for i in range(0xFF10, 0xFF1A)])
        
        # Гомоглифы - символы, похожие на латиницу/кириллицу
        self.homoglyphs = {
            'а': 'a',  # кириллическая 'а' похожа на латинскую 'a'
            'е': 'e',  # кириллическая 'е' похожа на латинскую 'e'
            'о': 'o',  # кириллическая 'о' похожа на латинскую 'o'
            'р': 'p',  # кириллическая 'р' похожа на латинскую 'p'
            'с': 'c',  # кириллическая 'с' похожа на латинскую 'c'
            'у': 'y',  # кириллическая 'у' похожа на латинскую 'y'
            'х': 'x',  # кириллическая 'х' похожа на латинскую 'x'
            'А': 'A',  # заглавные
            'В': 'B',
            'Е': 'E',
            'К': 'K',
            'М': 'M',
            'Н': 'H',
            'О': 'O',
            'Р': 'P',
            'С': 'C',
            'Т': 'T',
            'Х': 'X',
        }
        
        # Все подозрительные символы для быстрого поиска
        self.all_suspicious = self.sup_sub_chars.union(self.fullwidth_digits)
    
    def analyze(self, text: str) -> Dict:
        """Анализирует текст на наличие Unicode-артефактов"""
        results = {
            'sup_sub_count': 0,
            'sup_sub_positions': [],
            'fullwidth_count': 0,
            'fullwidth_positions': [],
            'homoglyph_count': 0,
            'homoglyph_examples': [],
            'density_per_10k': 0,
            'suspicious_chunks': []
        }
        
        total_chars = len(text)
        
        for i, char in enumerate(text):
            # Проверка надстрочных/подстрочных
            if char in self.sup_sub_chars:
                results['sup_sub_count'] += 1
                results['sup_sub_positions'].append(i)
                # Сохраняем контекст (окружающие символы)
                context = text[max(0, i-20):min(len(text), i+20)]
                results['suspicious_chunks'].append({
                    'char': char,
                    'position': i,
                    'context': context,
                    'type': 'superscript/subscript'
                })
            
            # Проверка fullwidth цифр
            elif char in self.fullwidth_digits:
                results['fullwidth_count'] += 1
                results['fullwidth_positions'].append(i)
                context = text[max(0, i-20):min(len(text), i+20)]
                results['suspicious_chunks'].append({
                    'char': char,
                    'position': i,
                    'context': context,
                    'type': 'fullwidth digit'
                })
            
            # Проверка гомоглифов (кириллица вместо латиницы или наоборот)
            # Упрощенная проверка - ищем кириллические символы в английском тексте
            if char in self.homoglyphs and i > 0 and i < len(text)-1:
                # Проверяем, что вокруг латиница или цифры (признак английского текста)
                surrounding = text[max(0, i-3):i] + text[i+1:min(len(text), i+4)]
                if all(ord(c) < 128 or c.isspace() or c in '.,;:!?' for c in surrounding):
                    results['homoglyph_count'] += 1
                    results['homoglyph_examples'].append({
                        'char': char,
                        'looks_like': self.homoglyphs[char],
                        'position': i
                    })
        
        # Плотность на 10,000 символов
        total_suspicious = results['sup_sub_count'] + results['fullwidth_count']
        if total_chars > 0:
            results['density_per_10k'] = (total_suspicious * 10000) / total_chars
        
        # Оценка риска
        if results['density_per_10k'] > 8:
            results['risk_level'] = 'high'
            results['risk_score'] = 3
        elif results['density_per_10k'] > 3:
            results['risk_level'] = 'medium'
            results['risk_score'] = 2
        elif results['density_per_10k'] > 0:
            results['risk_level'] = 'low'
            results['risk_score'] = 1
        else:
            results['risk_level'] = 'none'
            results['risk_score'] = 0
        
        return results


class DashAnalyzer:
    """Анализ множественных длинных тире (уровень 1)"""
    
    def __init__(self):
        self.em_dash = '—'  # U+2014
    
    def analyze(self, sentences: List[str]) -> Dict:
        """Анализирует предложения на наличие множественных тире"""
        results = {
            'total_sentences': len(sentences),
            'sentences_with_multiple_dashes': [],
            'heavy_sentences': [],  # >=3 тире или >=2 в длинных предложениях
            'dash_counts_per_sentence': [],
            'percentage_heavy': 0,
            'examples': []
        }
        
        for sent in sentences:
            dash_count = sent.count(self.em_dash)
            results['dash_counts_per_sentence'].append(dash_count)
            
            word_count = len(sent.split())
            is_heavy = False
            
            if dash_count >= 3:
                is_heavy = True
            elif dash_count >= 2 and word_count > 90:
                is_heavy = True
            
            if dash_count >= 2:
                results['sentences_with_multiple_dashes'].append({
                    'sentence': sent[:100] + '...' if len(sent) > 100 else sent,
                    'dash_count': dash_count,
                    'word_count': word_count
                })
            
            if is_heavy:
                results['heavy_sentences'].append({
                    'sentence': sent[:100] + '...' if len(sent) > 100 else sent,
                    'dash_count': dash_count,
                    'word_count': word_count
                })
                if len(results['examples']) < 3:  # Сохраняем примеры
                    results['examples'].append(sent[:150])
        
        if len(sentences) > 0:
            results['percentage_heavy'] = (len(results['heavy_sentences']) / len(sentences)) * 100
        
        # Оценка риска
        if results['percentage_heavy'] > 5:
            results['risk_level'] = 'high'
            results['risk_score'] = 3
        elif results['percentage_heavy'] > 2:
            results['risk_level'] = 'medium'
            results['risk_score'] = 2
        elif results['percentage_heavy'] > 0:
            results['risk_level'] = 'low'
            results['risk_score'] = 1
        else:
            results['risk_level'] = 'none'
            results['risk_score'] = 0
        
        return results


class AIPhraseDetector:
    """Детектор характерных ИИ-фраз и штампов (уровень 1)"""
    
    def __init__(self):
        # Основные ИИ-фразы (обновляемый список)
        self.ai_phrases = [
            # "Модные" слова 2024-2025
            'delve into', 'testament to', 'pivotal role', 'sheds light',
            'in the tapestry', 'in the realm', 'underscores the', 'harnesses the',
            'ever-evolving landscape', 'nuanced understanding', 'robust framework',
            'holistic approach', 'paradigm shift', 'cutting-edge',
            
            # Устойчивые связки
            'it is worth noting that', 'it is important to note that',
            'it should be noted that', 'as we delve deeper',
            'in the context of', 'with respect to', 'in terms of',
            'on the one hand', 'on the other hand',
            
            # Избыточные переходы
            'moreover', 'furthermore', 'in addition', 'consequently',
            'therefore', 'thus', 'hence', 'nonetheless', 'nevertheless',
            'accordingly', 'as a result', 'for this reason',
            
            # Академические клише
            'a wide range of', 'a variety of', 'numerous studies',
            'previous research', 'existing literature', 'prior work',
            'recent advances', 'state-of-the-art', 'best of our knowledge',
        ]
        
        # Частотные пороги (на 1000 предложений)
        self.transition_threshold = 12
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует текст на наличие ИИ-фраз"""
        results = {
            'phrase_counts': {},
            'top_phrases': [],
            'transition_count': 0,
            'transition_density': 0,
            'repeated_phrases': [],
            'risk_score': 0,
            'risk_level': 'none'
        }
        
        text_lower = text.lower()
        
        # Подсчет каждой фразы
        for phrase in self.ai_phrases:
            count = text_lower.count(phrase.lower())
            if count > 0:
                results['phrase_counts'][phrase] = count
        
        # Топ-10 самых частых фраз
        if results['phrase_counts']:
            sorted_phrases = sorted(results['phrase_counts'].items(), 
                                   key=lambda x: x[1], reverse=True)
            results['top_phrases'] = sorted_phrases[:10]
            
            # Отмечаем фразы, которые встречаются слишком часто
            for phrase, count in sorted_phrases:
                if count > 4:  # Порог повторов
                    results['repeated_phrases'].append({
                        'phrase': phrase,
                        'count': count
                    })
        
        # Подсчет переходных конструкций
        transitions = ['moreover', 'furthermore', 'in addition', 
                      'consequently', 'therefore', 'thus', 'hence']
        for trans in transitions:
            results['transition_count'] += text_lower.count(trans)
        
        if len(sentences) > 0:
            results['transition_density'] = (results['transition_count'] * 1000) / len(sentences)
        
        # Оценка риска
        risk_score = 0
        
        # Риск от повторяющихся фраз
        if len(results['repeated_phrases']) > 2:
            risk_score += 2
        elif len(results['repeated_phrases']) > 0:
            risk_score += 1
        
        # Риск от переходных конструкций
        if results['transition_density'] > self.transition_threshold:
            risk_score += 2
        elif results['transition_density'] > self.transition_threshold * 0.7:
            risk_score += 1
        
        # Общая оценка
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        results['risk_score'] = risk_score
        
        return results


class BurstinessAnalyzer:
    """Анализ вариативности длины предложений (уровень 2)"""
    
    def analyze(self, sentences: List[str]) -> Dict:
        """Анализирует burstiness текста"""
        results = {}
        
        if len(sentences) < 5:
            return {'error': 'Too few sentences for analysis'}
        
        # Длина предложений в словах
        sent_lengths = [len(sent.split()) for sent in sentences]
        results['sentence_lengths'] = sent_lengths
        
        # Основные статистики
        results['mean_length'] = np.mean(sent_lengths)
        results['std_length'] = np.std(sent_lengths)
        results['min_length'] = np.min(sent_lengths)
        results['max_length'] = np.max(sent_lengths)
        results['median_length'] = np.median(sent_lengths)
        
        # Коэффициент вариации (CV)
        if results['mean_length'] > 0:
            results['cv'] = results['std_length'] / results['mean_length']
        else:
            results['cv'] = 0
        
        # Межквартильный размах (IQR)
        q75, q25 = np.percentile(sent_lengths, [75, 25])
        results['iqr'] = q75 - q25
        
        # Yule's characteristic (мера дисперсии)
        freq_dist = Counter(sent_lengths)
        N = len(sent_lengths)
        sum_fi2 = sum(freq ** 2 for freq in freq_dist.values())
        results['yule_characteristic'] = (sum_fi2 / N) - (1 / N) if N > 0 else 0
        
        # Оценка burstiness
        if results['cv'] < 0.35:
            results['burstiness'] = 'very_low'
            results['risk_level'] = 'high'
            results['risk_score'] = 3
        elif results['cv'] < 0.5:
            results['burstiness'] = 'low'
            results['risk_level'] = 'medium'
            results['risk_score'] = 2
        elif results['cv'] < 0.7:
            results['burstiness'] = 'normal'
            results['risk_level'] = 'low'
            results['risk_score'] = 1
        else:
            results['burstiness'] = 'high'
            results['risk_level'] = 'none'
            results['risk_score'] = 0
        
        return results


class PerplexityAnalyzer:
    """Анализ перплексии с помощью языковой модели (уровень 3)"""
    
    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = None
        self.model = None
        self.initialized = False
    
    @st.cache_resource
    def _load_model(_self):
        """Загружает модель (кэшируется Streamlit)"""
        try:
            _self.tokenizer = AutoTokenizer.from_pretrained(_self.model_name)
            _self.model = AutoModelForCausalLM.from_pretrained(
                _self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            if torch.cuda.is_available():
                _self.model = _self.model.cuda()
            _self.initialized = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            _self.initialized = False
        return _self
    
    def analyze(self, text: str, chunk_size: int = 512) -> Dict:
        """Анализирует перплексию текста"""
        results = {
            'perplexities': [],
            'mean_perplexity': 0,
            'median_perplexity': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'error': None
        }
        
        if not self.initialized:
            self._load_model()
        
        if not self.initialized:
            results['error'] = "Model not initialized"
            return results
        
        # Разбиваем на чанки
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        # Вычисляем перплексию для каждого чанка
        for chunk in chunks[:5]:  # Ограничиваем для скорости
            try:
                inputs = self.tokenizer(chunk, return_tensors='pt', 
                                      truncation=True, max_length=chunk_size)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    results['perplexities'].append(perplexity)
            except Exception as e:
                continue
        
        if results['perplexities']:
            results['mean_perplexity'] = np.mean(results['perplexities'])
            results['median_perplexity'] = np.median(results['perplexities'])
            
            # Оценка риска (пороги для TinyLlama)
            if results['mean_perplexity'] < 25:
                results['risk_level'] = 'high'
                results['risk_score'] = 3
            elif results['mean_perplexity'] < 45:
                results['risk_level'] = 'medium'
                results['risk_score'] = 2
            elif results['mean_perplexity'] < 70:
                results['risk_level'] = 'low'
                results['risk_score'] = 1
            else:
                results['risk_level'] = 'none'
                results['risk_score'] = 0
        
        return results


class GrammarAnalyzer:
    """Анализ грамматических особенностей (пассив, номинализация) (уровень 2)"""
    
    def __init__(self):
        self.nlp = None
        self._load_spacy()
    
    @st.cache_resource
    def _load_spacy(_self):
        """Загружает spacy модель"""
        try:
            # Пробуем загрузить модель
            _self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Если модель не найдена, скачиваем
            with st.spinner("Скачивание модели spaCy... Это может занять минуту"):
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                    capture_output=True
                )
            _self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Ошибка загрузки spaCy модели: {e}")
            # Создаем минимальный пайплайн без модели
            _self.nlp = spacy.blank("en")
        return _self

    def analyze(self, text: str) -> Dict:
        """Анализирует грамматические особенности"""
        results = {
            'passive_count': 0,
            'passive_sentences': 0,
            'nominalization_count': 0,
            'total_sentences': 0,
            'passive_percentage': 0,
            'nominalizations_per_1000': 0,
            'examples': [],
            'risk_level': 'none',
            'risk_score': 0
        }
        
        # Проверяем, загружена ли модель
        if self.nlp is None:
            self._load_spacy()
        
        # Если модель все еще None, возвращаем пустой результат
        if self.nlp is None:
            results['error'] = "spaCy model not available"
            return results
        
        try:
            # Обрабатываем текст с ограничением по размеру
            text_sample = text[:10000] if len(text) > 10000 else text
            doc = self.nlp(text_sample)
            
            # Суффиксы номинализации
            nominalization_suffixes = ['tion', 'ment', 'ance', 'ence', 'ing', 'al', 'ity', 'ism']
            
            for sent in doc.sents:
                results['total_sentences'] += 1
                
                # Поиск пассивных конструкций
                has_passive = False
                for token in sent:
                    # Пассив: быть + причастие прошедшего времени
                    if token.dep_ == 'auxpass' or (token.tag_ == 'VBN' and any(t.dep_ == 'auxpass' for t in token.head.children)):
                        has_passive = True
                        
                if has_passive:
                    results['passive_sentences'] += 1
                    if len(results['examples']) < 3:
                        results['examples'].append(str(sent)[:150])
            
            # Поиск номинализаций
            for token in doc:
                if token.pos_ == 'NOUN':
                    for suffix in nominalization_suffixes:
                        if token.text.lower().endswith(suffix):
                            results['nominalization_count'] += 1
                            break
            
            if results['total_sentences'] > 0:
                results['passive_percentage'] = (results['passive_sentences'] / results['total_sentences']) * 100
            
            total_words = len([t for t in doc if t.is_alpha])
            if total_words > 0:
                results['nominalizations_per_1000'] = (results['nominalization_count'] * 1000) / total_words
            
            # Оценка риска
            risk_score = 0
            if results['passive_percentage'] > 45:
                risk_score += 2
            elif results['passive_percentage'] > 35:
                risk_score += 1
            
            if results['nominalizations_per_1000'] > 25:
                risk_score += 2
            elif results['nominalizations_per_1000'] > 15:
                risk_score += 1
            
            if risk_score >= 3:
                results['risk_level'] = 'high'
            elif risk_score >= 2:
                results['risk_level'] = 'medium'
            elif risk_score >= 1:
                results['risk_level'] = 'low'
            
            results['risk_score'] = risk_score
            
        except Exception as e:
            results['error'] = str(e)
            results['risk_level'] = 'error'
        
        return results

class SemanticAnalyzer:
    """Анализ семантической близости предложений (уровень 3)"""
    
    def __init__(self):
        self.model = None
    
    @st.cache_resource
    def _load_model(_self):
        """Загружает sentence-transformer модель"""
        _self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return _self
    
    def analyze(self, sentences: List[str], max_sentences: int = 50) -> Dict:
        """Анализирует семантическую близость предложений"""
        results = {
            'similarities': [],
            'mean_similarity': 0,
            'median_similarity': 0,
            'similarity_std': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'error': None
        }
        
        if len(sentences) < 5:
            results['error'] = "Too few sentences for analysis"
            return results
        
        if not self.model:
            self._load_model()
        
        # Берем не больше max_sentences для производительности
        sentences_sample = sentences[:max_sentences]
        
        try:
            # Получаем эмбеддинги
            embeddings = self.model.encode(sentences_sample, convert_to_tensor=True)
            
            # Вычисляем косинусную близость между соседними предложениями
            for i in range(len(embeddings) - 1):
                sim = torch.cosine_similarity(embeddings[i].unsqueeze(0), 
                                            embeddings[i+1].unsqueeze(0))
                results['similarities'].append(sim.item())
            
            if results['similarities']:
                results['mean_similarity'] = np.mean(results['similarities'])
                results['median_similarity'] = np.median(results['similarities'])
                results['similarity_std'] = np.std(results['similarities'])
                
                # Оценка риска (высокая близость = гладкий текст = вероятно ИИ)
                if results['mean_similarity'] > 0.75:
                    results['risk_level'] = 'high'
                    results['risk_score'] = 3
                elif results['mean_similarity'] > 0.6:
                    results['risk_level'] = 'medium'
                    results['risk_score'] = 2
                elif results['mean_similarity'] > 0.45:
                    results['risk_level'] = 'low'
                    results['risk_score'] = 1
                else:
                    results['risk_level'] = 'none'
                    results['risk_score'] = 0
                    
        except Exception as e:
            results['error'] = str(e)
        
        return results


class HedgingAnalyzer:
    """Анализ хеджинга (слов неуверенности) (уровень 2)"""
    
    def __init__(self):
        self.hedging_words = [
            'may', 'might', 'could', 'would', 'should',
            'possibly', 'probably', 'perhaps', 'maybe',
            'likely', 'unlikely', 'potentially',
            'appears', 'appear', 'apparently',
            'seems', 'seem', 'seemingly',
            'suggests', 'suggest', 'suggestion',
            'indicates', 'indicate', 'indication',
            'implies', 'imply', 'implication',
            'generally', 'typically', 'usually',
            'often', 'sometimes', 'frequently',
            'to some extent', 'in some cases',
            'to a certain degree', 'relatively',
            'comparatively', 'approximately'
        ]
        
        self.personal_pronouns = ['we', 'our', 'us', 'i', 'my', 'mine']
        
    def analyze(self, text: str) -> Dict:
        """Анализирует использование хеджинга и личных местоимений"""
        results = {
            'hedging_count': 0,
            'hedging_per_1000': 0,
            'personal_count': 0,
            'personal_per_1000': 0,
            'total_words': 0,
            'risk_level': 'none',
            'risk_score': 0
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        results['total_words'] = len(words)
        
        # Подсчет хеджинга
        for word in self.hedging_words:
            if ' ' in word:  # Фразы
                results['hedging_count'] += text_lower.count(word)
            else:  # Отдельные слова
                results['hedging_count'] += sum(1 for w in words if w == word)
        
        # Подсчет личных местоимений
        for pronoun in self.personal_pronouns:
            results['personal_count'] += sum(1 for w in words if w == pronoun)
        
        # Нормировка на 1000 слов
        if results['total_words'] > 0:
            results['hedging_per_1000'] = (results['hedging_count'] * 1000) / results['total_words']
            results['personal_per_1000'] = (results['personal_count'] * 1000) / results['total_words']
        
        # Оценка риска (низкий хеджинг = категоричность = вероятно ИИ)
        risk_score = 0
        if results['hedging_per_1000'] < 3:
            risk_score += 3
        elif results['hedging_per_1000'] < 5:
            risk_score += 2
        elif results['hedging_per_1000'] < 7:
            risk_score += 1
        
        # В научных статьях ожидается некоторое количество личных местоимений
        if results['personal_per_1000'] < 0.5:
            risk_score += 1
        
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        results['risk_score'] = risk_score
        
        return results


class DocumentProcessor:
    """Обработчик загруженных документов"""
    
    @staticmethod
    def read_docx(file) -> str:
        """Читает .docx файл"""
        doc = Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        # Читаем таблицы
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        full_text.append(para.text)
        
        return '\n'.join(full_text)
    
    @staticmethod
    def read_doc(file) -> Optional[str]:
        """Читает .doc файл (требует antiword)"""
        try:
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            # Используем antiword для извлечения текста
            result = subprocess.run(['antiword', tmp_path], 
                                   capture_output=True, text=True)
            os.unlink(tmp_path)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return None
        except:
            return None
    
    @staticmethod
    def split_sentences(text: str, nlp) -> List[str]:
        """Разбивает текст на предложения"""
        sentences = []
        try:
            # Разбиваем на части для обработки
            chunk_size = 50000
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            for chunk in text_chunks:
                doc = nlp(chunk)
                sentences.extend([str(sent).strip() for sent in doc.sents])
        except Exception as e:
            # Если не удалось разбить с помощью spacy, используем простой split
            sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 20]
        
        return sentences
    
    @staticmethod
    def preprocess(text: str) -> str:
        """Базовая предобработка текста"""
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        # Удаление лишних переносов строк
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


# ============================================================================
# Основное приложение
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🔬 AI Detector for Scientific Papers</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Загрузите научную статью в формате .docx или .doc для многоуровневого анализа 
    на предмет использования искусственного интеллекта при написании.
    </div>
    """, unsafe_allow_html=True)
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки анализа")
        
        analysis_depth = st.select_slider(
            "Глубина анализа",
            options=['Быстрый', 'Стандартный', 'Глубокий'],
            value='Стандартный'
        )
        
        st.markdown("---")
        st.markdown("**Активные модули:**")
        
        modules = {
            'unicode': st.checkbox("Unicode-артефакты", value=True),
            'dashes': st.checkbox("Множественные тире", value=True),
            'phrases': st.checkbox("ИИ-фразы", value=True),
            'burstiness': st.checkbox("Burstiness", value=True),
            'grammar': st.checkbox("Грамматика", value=False),
            'perplexity': st.checkbox("Perplexity (медленно)", value=False),
            'semantic': st.checkbox("Семантика (медленно)", value=False),
            'hedging': st.checkbox("Хеджинг", value=True)
        }
        
        st.markdown("---")
        st.markdown("**Пороги чувствительности:**")
        sensitivity = st.select_slider(
            "",
            options=['Низкая', 'Средняя', 'Высокая'],
            value='Средняя'
        )
        
        st.markdown("---")
        st.markdown("**Загрузка модели spaCy**")
        if st.button("Загрузить/обновить модель"):
            with st.spinner("Загрузка модели..."):
                os.system("python -m spacy download en_core_web_sm")
            st.success("Модель загружена!")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите файл статьи", 
        type=['docx', 'doc'],
        help="Поддерживаются форматы .docx и .doc"
    )
    
    if uploaded_file is not None:
        # Показываем информацию о файле
        file_details = {
            "Имя файла": uploaded_file.name,
            "Тип": uploaded_file.type,
            "Размер": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("📄 Информация о файле:", file_details)
        
        # Прогресс
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Шаг 1: Чтение документа
            status_text.text("Чтение документа...")
            progress_bar.progress(10)
            
            if uploaded_file.name.endswith('.docx'):
                text = DocumentProcessor.read_docx(uploaded_file)
            else:  # .doc
                text = DocumentProcessor.read_doc(uploaded_file)
                if text is None:
                    st.error("""
                    Не удалось прочитать .doc файл. Убедитесь, что установлен antiword:
                    - Ubuntu/Debian: sudo apt-get install antiword
                    - MacOS: brew install antiword
                    """)
                    return
            
            if not text or len(text.strip()) < 100:
                st.warning("Файл слишком мал или не содержит текста")
                return
            
            # Шаг 2: Предобработка
            status_text.text("Предобработка текста...")
            progress_bar.progress(20)
            text = DocumentProcessor.preprocess(text)
            
            # Шаг 3: Загрузка spacy для сегментации
            status_text.text("Загрузка NLP модели...")
            progress_bar.progress(30)
            
            try:
                # Пробуем загрузить модель
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Если модель не найдена, пытаемся скачать
                status_text.text("Скачивание NLP модели...")
                import subprocess
                try:
                    subprocess.run(
                        ["python", "-m", "spacy", "download", "en_core_web_sm"],
                        check=True,
                        capture_output=True
                    )
                    nlp = spacy.load("en_core_web_sm")
                except:
                    st.warning("Не удалось загрузить полную модель spaCy. Используется упрощенная версия.")
                    # Создаем пустой пайплайн для базовой обработки
                    nlp = spacy.blank("en")
            except Exception as e:
                st.warning(f"Ошибка загрузки spaCy: {e}. Используется упрощенная обработка.")
                nlp = spacy.blank("en")
            
            # Шаг 4: Сегментация на предложения
            status_text.text("Сегментация текста...")
            progress_bar.progress(40)
            sentences = DocumentProcessor.split_sentences(text, nlp)
            
            st.success(f"✅ Текст загружен: {len(text)} символов, {len(sentences)} предложений")
            
            # Создаем вкладки для результатов
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Общий отчет", 
                "🔍 Артефакты", 
                "📈 Статистика",
                "🧠 Семантика",
                "📝 Текст"
            ])
            
            # Хранилище результатов
            results = {}
            risk_scores = []
            
            # =================================================================
            # Модуль 1: Unicode-артефакты
            # =================================================================
            if modules['unicode']:
                with st.spinner("Анализ Unicode-артефактов..."):
                    detector = UnicodeArtifactDetector()
                    results['unicode'] = detector.analyze(text)
                    risk_scores.append(results['unicode']['risk_score'])
                progress_bar.progress(50)
            
            # =================================================================
            # Модуль 2: Множественные тире
            # =================================================================
            if modules['dashes']:
                with st.spinner("Анализ множественных тире..."):
                    detector = DashAnalyzer()
                    results['dashes'] = detector.analyze(sentences)
                    risk_scores.append(results['dashes']['risk_score'])
                progress_bar.progress(55)
            
            # =================================================================
            # Модуль 3: ИИ-фразы
            # =================================================================
            if modules['phrases']:
                with st.spinner("Поиск ИИ-фраз..."):
                    detector = AIPhraseDetector()
                    results['phrases'] = detector.analyze(text, sentences)
                    risk_scores.append(results['phrases']['risk_score'])
                progress_bar.progress(60)
            
            # =================================================================
            # Модуль 4: Burstiness
            # =================================================================
            if modules['burstiness']:
                with st.spinner("Анализ вариативности предложений..."):
                    detector = BurstinessAnalyzer()
                    results['burstiness'] = detector.analyze(sentences)
                    if 'error' not in results['burstiness']:
                        risk_scores.append(results['burstiness']['risk_score'])
                progress_bar.progress(65)
            
            # =================================================================
            # Модуль 5: Грамматика (пассив/номинализация)
            # =================================================================
            if modules['grammar']:
                with st.spinner("Грамматический анализ..."):
                    detector = GrammarAnalyzer()
                    results['grammar'] = detector.analyze(text)
                    risk_scores.append(results['grammar']['risk_score'])
                progress_bar.progress(70)
            
            # =================================================================
            # Модуль 6: Хеджинг
            # =================================================================
            if modules['hedging']:
                with st.spinner("Анализ хеджинга..."):
                    detector = HedgingAnalyzer()
                    results['hedging'] = detector.analyze(text)
                    risk_scores.append(results['hedging']['risk_score'])
                progress_bar.progress(75)
            
            # =================================================================
            # Модуль 7: Perplexity (тяжелый)
            # =================================================================
            if modules['perplexity'] and analysis_depth in ['Стандартный', 'Глубокий']:
                with st.spinner("Анализ перплексии (может занять минуту)..."):
                    detector = PerplexityAnalyzer()
                    results['perplexity'] = detector.analyze(text)
                    if 'error' not in results['perplexity']:
                        risk_scores.append(results['perplexity']['risk_score'])
                progress_bar.progress(85)
            
            # =================================================================
            # Модуль 8: Семантическая близость (тяжелый)
            # =================================================================
            if modules['semantic'] and analysis_depth in ['Глубокий']:
                with st.spinner("Семантический анализ..."):
                    detector = SemanticAnalyzer()
                    results['semantic'] = detector.analyze(sentences)
                    if 'error' not in results['semantic']:
                        risk_scores.append(results['semantic']['risk_score'])
                progress_bar.progress(95)
            
            # Завершение
            progress_bar.progress(100)
            status_text.text("Анализ завершен!")
            
            # =================================================================
            # Вкладка 1: Общий отчет
            # =================================================================
            with tab1:
                st.header("📊 Общая оценка")
                
                # Интегральный риск
                if risk_scores:
                    avg_risk = np.mean(risk_scores)
                    max_risk = np.max(risk_scores)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if avg_risk < 1:
                            st.markdown('<p class="success-text">● Низкая вероятность ИИ</p>', 
                                      unsafe_allow_html=True)
                        elif avg_risk < 2:
                            st.markdown('<p class="warning-text">● Средняя вероятность ИИ</p>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="danger-text">● Высокая вероятность ИИ</p>', 
                                      unsafe_allow_html=True)
                        st.metric("Средний риск", f"{avg_risk:.2f}/3")
                    
                    with col2:
                        st.metric("Максимальный риск", f"{max_risk:.2f}/3")
                    
                    with col3:
                        st.metric("Активных модулей", len(risk_scores))
                    
                    # Радарная диаграмма
                    categories = []
                    values = []
                    
                    for key in results:
                        if 'risk_score' in results[key]:
                            categories.append(key)
                            values.append(results[key]['risk_score'])
                    
                    if categories:
                        fig = go.Figure(data=go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            marker=dict(color='red' if avg_risk > 1.5 else 'orange' if avg_risk > 0.5 else 'green')
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 3]
                                )),
                            showlegend=False,
                            title="Профиль рисков"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Сводная таблица
                st.subheader("Сводка по метрикам")
                summary_data = []
                for key, data in results.items():
                    if isinstance(data, dict) and 'risk_level' in data:
                        summary_data.append({
                            'Метрика': key,
                            'Уровень риска': data['risk_level'],
                            'Оценка': data.get('risk_score', 0),
                            'Детали': str(data.get('density_per_10k', data.get('percentage_heavy', 
                                      data.get('mean_perplexity', data.get('cv', 'N/A')))))
                        })
                
                if summary_data:
                    df = pd.DataFrame(summary_data)
                    st.dataframe(df, use_container_width=True)
            
            # =================================================================
            # Вкладка 2: Артефакты
            # =================================================================
            with tab2:
                st.header("🔍 Артефакты форматирования")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'unicode' in results:
                        st.subheader("Unicode-артефакты")
                        u = results['unicode']
                        st.metric("Плотность на 10k символов", f"{u['density_per_10k']:.2f}")
                        st.metric("Надстрочные/подстрочные", u['sup_sub_count'])
                        st.metric("Fullwidth цифры", u['fullwidth_count'])
                        st.metric("Гомоглифы", u['homoglyph_count'])
                        
                        if u['suspicious_chunks']:
                            st.write("**Примеры:**")
                            for chunk in u['suspicious_chunks'][:3]:
                                st.code(f"...{chunk['context']}...")
                
                with col2:
                    if 'dashes' in results:
                        st.subheader("Множественные тире")
                        d = results['dashes']
                        st.metric("Предложений с ≥2 тире", len(d['sentences_with_multiple_dashes']))
                        st.metric("Тяжелые предложения", len(d['heavy_sentences']))
                        st.metric("Процент тяжелых", f"{d['percentage_heavy']:.2f}%")
                        
                        if d['examples']:
                            st.write("**Примеры:**")
                            for ex in d['examples'][:2]:
                                st.info(ex)
                
                if 'phrases' in results:
                    st.subheader("ИИ-фразы и штампы")
                    p = results['phrases']
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Плотность переходов", f"{p['transition_density']:.2f}/1000 предл.")
                    
                    with col4:
                        st.metric("Повторяющихся фраз", len(p['repeated_phrases']))
                    
                    if p['top_phrases']:
                        st.write("**Топ-5 фраз:**")
                        for phrase, count in p['top_phrases'][:5]:
                            st.write(f"- '{phrase}': {count} раз(а)")
            
            # =================================================================
            # Вкладка 3: Статистика
            # =================================================================
            with tab3:
                st.header("📈 Статистический анализ")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'burstiness' in results and 'error' not in results['burstiness']:
                        st.subheader("Burstiness")
                        b = results['burstiness']
                        st.metric("Ср. длина предложения", f"{b['mean_length']:.1f} слов")
                        st.metric("Станд. отклонение", f"{b['std_length']:.1f}")
                        st.metric("Коэф. вариации (CV)", f"{b['cv']:.3f}")
                        st.metric("IQR", f"{b['iqr']:.1f}")
                        
                        # Гистограмма длин предложений
                        fig = px.histogram(x=b['sentence_lengths'], nbins=30,
                                         title="Распределение длины предложений")
                        fig.update_layout(xaxis_title="Длина (слова)", 
                                        yaxis_title="Частота")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'grammar' in results:
                        st.subheader("Грамматика")
                        g = results['grammar']
                        st.metric("Пассивных предложений", f"{g['passive_percentage']:.1f}%")
                        st.metric("Номинализаций на 1000 слов", f"{g['nominalizations_per_1000']:.1f}")
                        
                        if g['examples']:
                            st.write("**Пример пассива:**")
                            st.info(g['examples'][0])
                    
                    if 'hedging' in results:
                        st.subheader("Хеджинг")
                        h = results['hedging']
                        st.metric("Хеджинг на 1000 слов", f"{h['hedging_per_1000']:.2f}")
                        st.metric("Личные местоимения на 1000", f"{h['personal_per_1000']:.2f}")
            
            # =================================================================
            # Вкладка 4: Семантика
            # =================================================================
            with tab4:
                st.header("🧠 Глубинный семантический анализ")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'perplexity' in results and 'error' not in results['perplexity']:
                        st.subheader("Perplexity")
                        pp = results['perplexity']
                        st.metric("Средняя перплексия", f"{pp['mean_perplexity']:.2f}")
                        st.metric("Медианная перплексия", f"{pp['median_perplexity']:.2f}")
                        
                        if pp['perplexities']:
                            fig = px.line(y=pp['perplexities'], 
                                        title="Перплексия по чанкам")
                            fig.update_layout(xaxis_title="Чанк", 
                                            yaxis_title="Perplexity")
                            st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'semantic' in results and 'error' not in results['semantic']:
                        st.subheader("Семантическая близость")
                        sm = results['semantic']
                        st.metric("Ср. близость соседей", f"{sm['mean_similarity']:.3f}")
                        st.metric("Медианная близость", f"{sm['median_similarity']:.3f}")
                        st.metric("Стд. отклонение", f"{sm['similarity_std']:.3f}")
                        
                        if sm['similarities']:
                            fig = px.histogram(x=sm['similarities'], nbins=20,
                                             title="Распределение близости")
                            fig.update_layout(xaxis_title="Косинусная близость", 
                                            yaxis_title="Частота")
                            st.plotly_chart(fig, use_container_width=True)
            
            # =================================================================
            # Вкладка 5: Текст
            # =================================================================
            with tab5:
                st.header("📝 Исходный текст")
                
                # Добавляем подсветку подозрительных мест
                text_to_show = text[:5000] + "..." if len(text) > 5000 else text
                
                # Подсвечиваем длинные тире
                if 'dashes' in results and results['dashes']['heavy_sentences']:
                    st.subheader("⚠️ Подозрительные предложения (с множественными тире)")
                    for ex in results['dashes']['examples'][:3]:
                        st.markdown(f"> {ex}")
                
                st.subheader("Текст (первые 5000 символов)")
                st.text_area("", text_to_show, height=400)
        
        except Exception as e:
            st.error(f"Ошибка при обработке: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Информация о приложении
        st.info("👆 Загрузите файл для начала анализа")
        
        with st.expander("ℹ️ О метриках"):
            st.markdown("""
            ### Уровни анализа:
            
            **Уровень 1: Артефакты**
            - Unicode-гомоглифы: надстрочные/подстрочные символы, fullwidth цифры
            - Множественные длинные тире: ≥3 в предложении или ≥2 в предложениях >90 слов
            - ИИ-фразы: характерные для LLM выражения и штампы
            
            **Уровень 2: Статистика**
            - Burstiness: вариативность длины предложений
            - Пассив и номинализация: грамматические особенности
            - Хеджинг: использование слов неуверенности
            
            **Уровень 3: Семантика**
            - Perplexity: предсказуемость текста для языковой модели
            - Семантическая близость: косинусное расстояние между предложениями
            """)


if __name__ == "__main__":

    main()
