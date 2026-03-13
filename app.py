"""
Streamlit приложение для анализа научных статей на предмет использования ИИ
Исправленная версия с учетом совместимости версий и оптимизацией для Streamlit Cloud
Версия 3.0 - Добавлены: анализ скобок, пунктуации, апострофов, перечислений, полный дашборд
"""

import streamlit as st
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import tempfile
import os
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

# Фикс для совместимости pydantic v1 и v2 (нужно для spacy)
import pydantic
if hasattr(pydantic, 'v1'):
    # Если установлена pydantic v2, используем v1 совместимость
    from pydantic.v1 import BaseModel
else:
    # Если установлена pydantic v1
    from pydantic import BaseModel

# NLP библиотеки с обработкой ошибок импорта
try:
    # Подавляем ошибки pydantic в spacy
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='spacy')
        import spacy
        SPACY_AVAILABLE = True
except ImportError as e:
    SPACY_AVAILABLE = False
    st.error(f"Ошибка загрузки spaCy: {e}. Некоторые функции будут недоступны.")
except Exception as e:
    # Ловим любые другие ошибки при импорте spacy
    SPACY_AVAILABLE = False
    st.warning(f"Предупреждение при загрузке spaCy: {type(e).__name__}. Некоторые функции могут быть недоступны.")

# Для transformers - загружаем с обработкой ошибок
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Для sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Конфигурация страницы
st.set_page_config(
    page_title="AI Detector for Scientific Papers",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация - обновленная палитра "Дашборд исследователя"
st.markdown("""
<style>
    /* Основные цвета */
    :root {
        --primary-bg: #F8FAFC;
        --accent-dark-blue: #0F2B4B;
        --success-green: #2D6A4F;
        --warning-amber: #B45309;
        --danger-burgundy: #991B1B;
        --text-dark: #1E293B;
        --grid-light: #E2E8F0;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: var(--accent-dark-blue);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid var(--grid-light);
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 5px solid var(--accent-dark-blue);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .warning-text {
        color: var(--warning-amber);
        font-weight: bold;
    }
    
    .danger-text {
        color: var(--danger-burgundy);
        font-weight: bold;
    }
    
    .success-text {
        color: var(--success-green);
        font-weight: bold;
    }
    
    .info-box {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid var(--grid-light);
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--accent-dark-blue);
    }
    
    .risk-high {
        background-color: #FEF2F2;
        border-left: 5px solid var(--danger-burgundy);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .risk-medium {
        background-color: #FFFBEB;
        border-left: 5px solid var(--warning-amber);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .risk-low {
        background-color: #F0FDF4;
        border-left: 5px solid var(--success-green);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Стили для вкладок */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-dark);
        font-weight: 500;
    }
    
    /* Правая панель */
    .css-1d391kg {
        background-color: white;
        border-left: 1px solid var(--grid-light);
    }
    
    /* Кнопки */
    .stButton button {
        background-color: var(--accent-dark-blue);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton button:hover {
        background-color: #1a3a5c;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Классы и функции для анализа
# ============================================================================

class ReferenceCutoff:
    """Обрезка текста до раздела References"""
    
    @staticmethod
    def cut_at_references(text: str) -> str:
        """Обрезает текст до начала раздела со ссылками"""
        if not text:
            return text
        
        # Маркеры начала списка литературы (разные варианты)
        reference_markers = [
            r'\nreferences?\s*\n',
            r'\nbibliography\s*\n',
            r'\nliterature cited\s*\n',
            r'\nworks cited\s*\n',
            r'\ncited literature\s*\n',
            r'\nreference list\s*\n',
            r'\nлитература\s*\n',
            r'\nсписок литературы\s*\n',
            r'\nбиблиография\s*\n',
            r'\nreferences and notes\s*\n',
            r'\nreferences & notes\s*\n'
        ]
        
        # Ищем первый маркер
        text_lower = text.lower()
        cutoff_pos = len(text)
        
        for marker in reference_markers:
            # Ищем с учетом регистра
            match = re.search(marker, text_lower)
            if match:
                # Находим позицию в оригинальном тексте
                pos = text_lower.find(marker.strip())
                if 0 < pos < cutoff_pos:
                    cutoff_pos = pos
        
        # Если нашли маркер, обрезаем
        if cutoff_pos < len(text):
            return text[:cutoff_pos].strip()
        
        return text


class UnicodeArtifactDetector:
    """Детектор Unicode-артефактов (уровень 1)"""
    
    def __init__(self):
        # Надстрочные и подстрочные символы
        self.sup_sub_chars = set([
            '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',  # надстрочные цифры
            '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',  # подстрочные цифры
            '⁺', '⁻', '⁼', '⁽', '⁾',  # надстрочные знаки
            '₊', '₋', '₌', '₍', '₎',  # подстрочные знаки
            'ª', 'º',  # женский/мужской ординар
        ])
        
        # Fullwidth цифры
        self.fullwidth_digits = set([chr(i) for i in range(0xFF10, 0xFF1A)])
        
        # Гомоглифы - символы, похожие на латиницу/кириллицу
        self.homoglyphs = {
            'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
            'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
            'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X',
        }
        
        # Греческие символы (часто в научных статьях)
        self.greek_chars = set([
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
            'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
            'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
            'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω'
        ])
        
        # Кириллица
        self.cyrillic_chars = set([
            'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к',
            'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц',
            'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
            'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К',
            'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
            'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'
        ])
        
        # Арабские символы
        self.arabic_chars = set([
            'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س',
            'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م',
            'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ'
        ])
        
        # Все подозрительные символы
        self.all_suspicious = self.sup_sub_chars.union(self.fullwidth_digits)
    
    def analyze(self, text: str) -> Dict:
        """Анализирует текст на наличие Unicode-артефактов и нелатинских символов"""
        results = {
            'sup_sub_count': 0,
            'fullwidth_count': 0,
            'homoglyph_count': 0,
            'greek_count': 0,
            'cyrillic_count': 0,
            'arabic_count': 0,
            'non_latin_total': 0,
            'non_latin_per_1000': 0,
            'density_per_10k': 0,
            'suspicious_chunks': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            # Дополнительная статистика
            'statistics': {
                'mean_density': 0,
                'median_density': 0,
                'max_density': 0,
                'distribution': []
            }
        }
        
        total_chars = len(text)
        if total_chars == 0:
            return results
        
        # Считаем слова для нормировки
        words = text.split()
        total_words = len(words)
        
        # Проходим по тексту (полный анализ, без ограничения)
        for i, char in enumerate(text):
            # Проверка надстрочных/подстрочных
            if char in self.sup_sub_chars:
                results['sup_sub_count'] += 1
                context = text[max(0, i-20):min(len(text), i+20)]
                results['suspicious_chunks'].append({
                    'char': char,
                    'context': context,
                    'type': 'superscript/subscript'
                })
            
            # Проверка fullwidth цифр
            elif char in self.fullwidth_digits:
                results['fullwidth_count'] += 1
                context = text[max(0, i-20):min(len(text), i+20)]
                results['suspicious_chunks'].append({
                    'char': char,
                    'context': context,
                    'type': 'fullwidth digit'
                })
            
            # Проверка гомоглифов
            elif char in self.homoglyphs and i > 0 and i < len(text)-1:
                surrounding = text[max(0, i-3):i] + text[i+1:min(len(text), i+4)]
                if all(ord(c) < 128 or c.isspace() or c in '.,;:!?' for c in surrounding):
                    results['homoglyph_count'] += 1
            
            # Подсчет нелатинских символов
            if char in self.greek_chars:
                results['greek_count'] += 1
                results['non_latin_total'] += 1
            elif char in self.cyrillic_chars:
                results['cyrillic_count'] += 1
                results['non_latin_total'] += 1
            elif char in self.arabic_chars:
                results['arabic_count'] += 1
                results['non_latin_total'] += 1
        
        # Плотность на 10,000 символов
        total_suspicious = results['sup_sub_count'] + results['fullwidth_count']
        results['density_per_10k'] = (total_suspicious * 10000) / total_chars if total_chars > 0 else 0
        
        # Нелатинские символы на 1000 слов
        if total_words > 0:
            results['non_latin_per_1000'] = (results['non_latin_total'] * 1000) / total_words
        
        # Статистика распределения (разбиваем текст на сегменты по 1000 символов)
        segment_size = 1000
        segments = [text[i:i+segment_size] for i in range(0, len(text), segment_size)]
        densities = []
        
        for segment in segments:
            seg_suspicious = sum(1 for c in segment if c in self.all_suspicious)
            seg_density = (seg_suspicious * 10000) / len(segment) if len(segment) > 0 else 0
            densities.append(seg_density)
        
        if densities:
            results['statistics']['distribution'] = densities
            results['statistics']['mean_density'] = float(np.mean(densities))
            results['statistics']['median_density'] = float(np.median(densities))
            results['statistics']['max_density'] = float(np.max(densities))
        
        # Оценка риска (теперь вероятностная)
        risk_score = 0
        confidence = 0.5  # базовая уверенность
        
        # Учитываем плотность артефактов
        if results['density_per_10k'] > 8:
            risk_score += 3
            confidence = 0.9
        elif results['density_per_10k'] > 3:
            risk_score += 2
            confidence = 0.7
        elif results['density_per_10k'] > 0:
            risk_score += 1
            confidence = 0.6
        
        # Учитываем гомоглифы (более опасны)
        if results['homoglyph_count'] > 5:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['homoglyph_count'] > 0:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Очень много нелатинских символов (кроме греческих в научных) - подозрительно
        if results['non_latin_total'] > 100 and results['greek_count'] < results['non_latin_total'] * 0.8:
            risk_score += 1
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        # Преобразуем в уровни для обратной совместимости
        if risk_score >= 4:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score > 0:
            results['risk_level'] = 'low'
        
        # Ограничиваем количество chunks для отображения
        results['suspicious_chunks'] = results['suspicious_chunks'][:10]
        
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
            'heavy_sentences': [],
            'percentage_heavy': 0,
            'examples': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'mean_dashes_per_sentence': 0,
                'median_dashes_per_sentence': 0,
                'max_dashes_in_sentence': 0,
                'distribution': []
            }
        }
        
        if not sentences:
            return results
        
        dash_counts = []
        
        for sent in sentences:
            if not sent or len(sent.strip()) == 0:
                continue
                
            dash_count = sent.count(self.em_dash)
            dash_counts.append(dash_count)
            word_count = len(sent.split())
            is_heavy = False
            
            if dash_count >= 3:
                is_heavy = True
            elif dash_count >= 2 and word_count > 90:
                is_heavy = True
            
            if dash_count >= 2:
                sent_preview = sent[:100] + '...' if len(sent) > 100 else sent
                results['sentences_with_multiple_dashes'].append({
                    'sentence': sent_preview,
                    'dash_count': dash_count,
                    'word_count': word_count
                })
            
            if is_heavy:
                sent_preview = sent[:100] + '...' if len(sent) > 100 else sent
                results['heavy_sentences'].append({
                    'sentence': sent_preview,
                    'dash_count': dash_count,
                    'word_count': word_count
                })
                if len(results['examples']) < 5:
                    results['examples'].append(sent[:150])
        
        if sentences:
            results['percentage_heavy'] = (len(results['heavy_sentences']) / len(sentences)) * 100
            
            # Статистика распределения
            if dash_counts:
                results['statistics']['mean_dashes_per_sentence'] = float(np.mean(dash_counts))
                results['statistics']['median_dashes_per_sentence'] = float(np.median(dash_counts))
                results['statistics']['max_dashes_in_sentence'] = float(np.max(dash_counts))
                results['statistics']['distribution'] = dash_counts[:100]  # Первые 100 для графика
        
        # Вероятностная оценка риска
        risk_score = 0
        confidence = 0.5
        
        if results['percentage_heavy'] > 5:
            risk_score = 3
            confidence = 0.85
        elif results['percentage_heavy'] > 2:
            risk_score = 2
            confidence = 0.7
        elif results['percentage_heavy'] > 0:
            risk_score = 1
            confidence = 0.6
        
        # Если много предложений с тире, но не тяжелых - тоже подозрительно
        if len(results['sentences_with_multiple_dashes']) > len(sentences) * 0.1:
            risk_score = max(risk_score, 2)
            confidence = min(confidence + 0.1, 1.0)
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        # Ограничиваем количество для отображения
        results['sentences_with_multiple_dashes'] = results['sentences_with_multiple_dashes'][:15]
        results['heavy_sentences'] = results['heavy_sentences'][:10]
        
        return results


class AIPhraseDetector:
    """Детектор характерных ИИ-фраз и штампов (уровень 1)"""
    
    def __init__(self):
        # Основные ИИ-фразы (обновлено 2025-2026)
        self.ai_phrases = [
            # "Модные" слова 2024-2025
            'delve into', 'testament to', 'pivotal role', 'sheds light',
            'in the tapestry', 'in the realm', 'underscores', 'harnesses',
            'ever-evolving landscape', 'nuanced understanding', 'robust framework',
            'holistic approach', 'paradigm shift', 'cutting-edge',
            
            # Новые маркеры 2025-2026
            'crucial', 'pivotal', 'paramount', 'underscores', 'sheds light',
            'highlights the importance', 'testament to', 'integral', 'realm',
            'landscape', 'ever-evolving', 'tapestry', 'harness', 'delve',
            'delves', 'delving', 'intricate', 'meticulously', 'nuanced',
            'robust', 'unveiling', 'findings', 'revealed', 'demonstrated',
            'underscore', 'elucidate', 'illuminate', 'data-driven',
            'paves the way', 'leverage', 'leverages', 'leveraging',
            
            # Дополнительные ИИ-фразы
            'pathway', 'pathways',  # новые
            'signaling', 'signals',  # новые
            'Collectively',  # новые (с большой буквы)
            'manifest',  # новые
            'paradigm',  # уже есть, но оставляем
            
            # Устойчивые связки
            'it is worth noting', 'it is important to note',
            'it should be noted', 'as we delve deeper',
            'in the context of', 'with respect to', 'in terms of',
            'it is crucial to', 'it is paramount to',
            
            # Избыточные переходы
            'moreover', 'furthermore', 'in addition', 'consequently',
            'therefore', 'thus', 'hence', 'nonetheless', 'nevertheless',
            'accordingly', 'as a result', 'for this reason',
            
            # Усилители уверенности
            'significantly', 'substantially', 'dramatically',
            'remarkably', 'notably', 'strikingly', 'profoundly'
        ]
        
        # Метадискурс взаимодействия (контрастные связки)
        self.metadiscourse_markers = [
            'however', 'nevertheless', 'nonetheless', 'yet',
            'although', 'even though', 'despite', 'in spite of',
            'conversely', 'in contrast', 'on the contrary',
            'on the one hand', 'on the other hand'
        ]
        
        self.transition_threshold = 12
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует текст на наличие ИИ-фраз"""
        results = {
            'phrase_counts': {},
            'top_phrases': [],
            'metadiscourse_count': 0,
            'metadiscourse_per_1000': 0,
            'transition_count': 0,
            'transition_density': 0,
            'repeated_phrases': [],
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'statistics': {
                'mean_phrases_per_sentence': 0,
                'median_phrases_per_sentence': 0,
                'max_phrases_in_sentence': 0,
                'distribution': []
            }
        }
        
        if not text or not sentences:
            return results
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        # Подсчет каждой фразы
        for phrase in self.ai_phrases:
            count = text_lower.count(phrase.lower())
            if count > 0:
                results['phrase_counts'][phrase] = count
        
        # Топ-15 самых частых фраз
        if results['phrase_counts']:
            sorted_phrases = sorted(results['phrase_counts'].items(), 
                                   key=lambda x: x[1], reverse=True)
            results['top_phrases'] = sorted_phrases[:15]
            
            # Отмечаем фразы, которые встречаются слишком часто
            for phrase, count in sorted_phrases:
                if count > 5:
                    results['repeated_phrases'].append({
                        'phrase': phrase,
                        'count': count
                    })
        
        # Подсчет переходных конструкций
        transitions = ['moreover', 'furthermore', 'in addition', 
                      'consequently', 'therefore', 'thus', 'hence']
        for trans in transitions:
            results['transition_count'] += text_lower.count(trans)
        
        if total_words > 0:
            results['transition_density'] = (results['transition_count'] * 1000) / total_words
        
        # Подсчет метадискурса (важно для научных статей)
        for marker in self.metadiscourse_markers:
            results['metadiscourse_count'] += text_lower.count(marker)
        
        if total_words > 0:
            results['metadiscourse_per_1000'] = (results['metadiscourse_count'] * 1000) / total_words
        
        # Статистика по предложениям
        phrases_per_sentence = []
        for sent in sentences[:100]:  # Первые 100 предложений
            sent_lower = sent.lower()
            sent_phrase_count = sum(1 for phrase in self.ai_phrases if phrase.lower() in sent_lower)
            phrases_per_sentence.append(sent_phrase_count)
        
        if phrases_per_sentence:
            results['statistics']['mean_phrases_per_sentence'] = float(np.mean(phrases_per_sentence))
            results['statistics']['median_phrases_per_sentence'] = float(np.median(phrases_per_sentence))
            results['statistics']['max_phrases_in_sentence'] = float(np.max(phrases_per_sentence))
            results['statistics']['distribution'] = phrases_per_sentence
        
        # Вероятностная оценка риска
        risk_score = 0
        confidence = 0.5
        
        if len(results['repeated_phrases']) > 3:
            risk_score += 3
            confidence = min(confidence + 0.3, 1.0)
        elif len(results['repeated_phrases']) > 1:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif len(results['repeated_phrases']) > 0:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        if results['transition_density'] > self.transition_threshold:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['transition_density'] > self.transition_threshold * 0.7:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Низкий метадискурс - подозрительно для научной статьи
        if results['metadiscourse_per_1000'] < 2 and len(sentences) > 20:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 4:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        return results


class BurstinessAnalyzer:
    """Анализ вариативности длины предложений (уровень 2)"""
    
    def analyze(self, sentences: List[str]) -> Dict:
        """Анализирует burstiness текста"""
        results = {
            'sentence_lengths': [],
            'mean_length': 0,
            'std_length': 0,
            'cv': 0,
            'iqr': 0,
            'burstiness': 'unknown',
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'median_length': 0,
                'max_length': 0,
                'min_length': 0,
                'percentile_25': 0,
                'percentile_75': 0,
                'distribution': []
            }
        }
        
        if len(sentences) < 5:
            results['error'] = 'Too few sentences for analysis'
            return results
        
        # Длина предложений в словах
        sent_lengths = [len(sent.split()) for sent in sentences if len(sent.strip()) > 0]
        if not sent_lengths:
            return results
            
        results['sentence_lengths'] = sent_lengths
        
        # Основные статистики
        results['mean_length'] = float(np.mean(sent_lengths))
        results['std_length'] = float(np.std(sent_lengths))
        
        # Коэффициент вариации (CV)
        if results['mean_length'] > 0:
            results['cv'] = float(results['std_length'] / results['mean_length'])
        
        # Межквартильный размах (IQR)
        q75, q25 = np.percentile(sent_lengths, [75, 25])
        results['iqr'] = float(q75 - q25)
        
        # Дополнительная статистика
        results['statistics']['median_length'] = float(np.median(sent_lengths))
        results['statistics']['max_length'] = float(np.max(sent_lengths))
        results['statistics']['min_length'] = float(np.min(sent_lengths))
        results['statistics']['percentile_25'] = float(q25)
        results['statistics']['percentile_75'] = float(q75)
        results['statistics']['distribution'] = sent_lengths[:200]  # Первые 200 для графика
        
        # Вероятностная оценка burstiness
        risk_score = 0
        confidence = 0.5
        
        if results['cv'] < 0.35:
            results['burstiness'] = 'very_low'
            risk_score = 3
            confidence = 0.9
        elif results['cv'] < 0.5:
            results['burstiness'] = 'low'
            risk_score = 2
            confidence = 0.7
        elif results['cv'] < 0.7:
            results['burstiness'] = 'normal'
            risk_score = 1
            confidence = 0.5
        else:
            results['burstiness'] = 'high'
            risk_score = 0
            confidence = 0.3
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        return results


class GrammarAnalyzer:
    """Анализ грамматических особенностей (без spacy, упрощенная версия)"""
    
    def __init__(self):
        self.nominalization_suffixes = ['tion', 'ment', 'ance', 'ence', 'ing', 'ity', 'ism', 'sis', 'ure', 'age']
        
        # Модальные глаголы для hedging
        self.modal_verbs = ['may', 'might', 'could', 'would', 'should', 'can']
        
        # Epistemic markers
        self.epistemic_markers = [
            'seem', 'appear', 'suggest', 'indicate', 'likely', 'unlikely',
            'possibly', 'probably', 'perhaps', 'maybe', 'potentially',
            'presumably', 'arguably', 'tentatively'
        ]
        
        # Усилители уверенности
        self.certainty_boosters = [
            'crucial', 'pivotal', 'paramount', 'essential', 'vital',
            'undoubtedly', 'certainly', 'definitely', 'clearly', 'obviously',
            'demonstrates', 'proves', 'confirms', 'establishes'
        ]
    
    def analyze(self, text: str) -> Dict:
        """Упрощенный анализ грамматики без spacy"""
        results = {
            'passive_indicators': 0,
            'nominalization_count': 0,
            'modal_count': 0,
            'epistemic_count': 0,
            'certainty_boosters_count': 0,
            'total_sentences': 0,
            'passive_percentage': 0,
            'nominalizations_per_1000': 0,
            'modals_per_1000': 0,
            'epistemic_per_1000': 0,
            'boosters_per_1000': 0,
            'examples': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'mean_passive_per_sentence': 0,
                'median_passive_per_sentence': 0,
                'max_passive_in_sentence': 0,
                'distribution': []
            }
        }
        
        if not text:
            return results
        
        # Простая сегментация на предложения
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.strip()) > 10]
        results['total_sentences'] = len(valid_sentences)
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        # Поиск индикаторов пассива (was/were + ed/en)
        passive_pattern = r'\b(was|were|is|are|been|being)\s+(\w+ed|\w+en)\b'
        passive_matches = re.findall(passive_pattern, text_lower)
        results['passive_indicators'] = len(passive_matches)
        
        # Суффиксы номинализации
        for word in words:
            for suffix in self.nominalization_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    results['nominalization_count'] += 1
                    break
        
        # Модальные глаголы
        for modal in self.modal_verbs:
            results['modal_count'] += sum(1 for w in words if w == modal)
        
        # Epistemic markers
        for marker in self.epistemic_markers:
            if ' ' in marker:
                results['epistemic_count'] += text_lower.count(marker)
            else:
                results['epistemic_count'] += sum(1 for w in words if w == marker)
        
        # Усилители уверенности
        for booster in self.certainty_boosters:
            results['certainty_boosters_count'] += sum(1 for w in words if w == booster)
        
        # Нормировка на 1000 слов
        if total_words > 0:
            results['nominalizations_per_1000'] = (results['nominalization_count'] * 1000) / total_words
            results['modals_per_1000'] = (results['modal_count'] * 1000) / total_words
            results['epistemic_per_1000'] = (results['epistemic_count'] * 1000) / total_words
            results['boosters_per_1000'] = (results['certainty_boosters_count'] * 1000) / total_words
        
        if results['total_sentences'] > 0:
            results['passive_percentage'] = (results['passive_indicators'] / results['total_sentences']) * 100
        
        # Статистика по предложениям
        passive_per_sentence = []
        for sent in valid_sentences[:100]:
            sent_lower = sent.lower()
            sent_passive = len(re.findall(passive_pattern, sent_lower))
            passive_per_sentence.append(sent_passive)
        
        if passive_per_sentence:
            results['statistics']['mean_passive_per_sentence'] = float(np.mean(passive_per_sentence))
            results['statistics']['median_passive_per_sentence'] = float(np.median(passive_per_sentence))
            results['statistics']['max_passive_in_sentence'] = float(np.max(passive_per_sentence))
            results['statistics']['distribution'] = passive_per_sentence
        
        # Вероятностная оценка риска
        risk_score = 0
        confidence = 0.5
        
        # Пассив (слишком много - подозрительно для ИИ?)
        if results['passive_percentage'] > 40:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['passive_percentage'] > 30:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Номинализации (слишком много - академический стиль, но может быть ИИ)
        if results['nominalizations_per_1000'] > 25:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['nominalizations_per_1000'] > 15:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Модальные глаголы (слишком мало - недостаточно hedging)
        if results['modals_per_1000'] < 3 and total_words > 500:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['modals_per_1000'] < 5 and total_words > 500:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Epistemic markers (слишком мало)
        if results['epistemic_per_1000'] < 2 and total_words > 500:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Усилители уверенности (слишком много)
        if results['boosters_per_1000'] > 5 and total_words > 500:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 4:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        # Примеры пассива
        for match in passive_matches[:5]:
            results['examples'].append(f"...{' '.join(match)}...")
        
        return results


class HedgingAnalyzer:
    """Анализ хеджинга (слов неуверенности)"""
    
    def __init__(self):
        self.hedging_words = [
            'may', 'might', 'could', 'would', 'should',
            'possibly', 'probably', 'perhaps', 'maybe',
            'likely', 'unlikely', 'potentially',
            'appears', 'seems', 'suggests', 'indicates',
            'generally', 'typically', 'usually', 'often',
            'to some extent', 'in some cases', 'relatively',
            'somewhat', 'partially', 'tentatively', 'arguably'
        ]
        
        self.personal_pronouns = ['we', 'our', 'us', 'i', 'my', 'mine']
        
        # Категоричные выражения (анти-хеджинг)
        self.certainty_phrases = [
            'clearly', 'obviously', 'undoubtedly', 'certainly',
            'definitely', 'absolutely', 'without doubt',
            'it is clear that', 'it is obvious that', 'there is no doubt'
        ]
        
    def analyze(self, text: str) -> Dict:
        """Анализирует использование хеджинга"""
        results = {
            'hedging_count': 0,
            'hedging_per_1000': 0,
            'certainty_count': 0,
            'certainty_per_1000': 0,
            'personal_count': 0,
            'personal_per_1000': 0,
            'hedging_ratio': 0,  # hedging / certainty
            'total_words': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'mean_hedging_per_sentence': 0,
                'median_hedging_per_sentence': 0,
                'max_hedging_in_sentence': 0,
                'distribution': []
            }
        }
        
        if not text:
            return results
        
        text_lower = text.lower()
        words = text_lower.split()
        results['total_words'] = len(words)
        
        # Подсчет хеджинга
        for word in self.hedging_words:
            if ' ' in word:
                results['hedging_count'] += text_lower.count(word)
            else:
                results['hedging_count'] += sum(1 for w in words if w == word)
        
        # Подсчет категоричных выражений
        for phrase in self.certainty_phrases:
            if ' ' in phrase:
                results['certainty_count'] += text_lower.count(phrase)
            else:
                results['certainty_count'] += sum(1 for w in words if w == phrase)
        
        # Подсчет личных местоимений
        for pronoun in self.personal_pronouns:
            results['personal_count'] += sum(1 for w in words if w == pronoun)
        
        # Нормировка на 1000 слов
        if results['total_words'] > 0:
            results['hedging_per_1000'] = (results['hedging_count'] * 1000) / results['total_words']
            results['certainty_per_1000'] = (results['certainty_count'] * 1000) / results['total_words']
            results['personal_per_1000'] = (results['personal_count'] * 1000) / results['total_words']
            
            if results['certainty_count'] > 0:
                results['hedging_ratio'] = results['hedging_count'] / results['certainty_count']
        
        # Статистика по предложениям (простая)
        sentences = re.split(r'[.!?]+', text)
        hedging_per_sentence = []
        
        for sent in sentences[:100]:
            sent_lower = sent.lower()
            sent_hedging = sum(1 for word in self.hedging_words if word in sent_lower)
            hedging_per_sentence.append(sent_hedging)
        
        if hedging_per_sentence:
            results['statistics']['mean_hedging_per_sentence'] = float(np.mean(hedging_per_sentence))
            results['statistics']['median_hedging_per_sentence'] = float(np.median(hedging_per_sentence))
            results['statistics']['max_hedging_in_sentence'] = float(np.max(hedging_per_sentence))
            results['statistics']['distribution'] = hedging_per_sentence
        
        # Вероятностная оценка риска
        risk_score = 0
        confidence = 0.5
        
        # Слишком мало хеджинга - главный маркер
        if results['hedging_per_1000'] < 3:
            risk_score += 3
            confidence = min(confidence + 0.3, 1.0)
        elif results['hedging_per_1000'] < 5:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['hedging_per_1000'] < 7:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Слишком много категоричности
        if results['certainty_per_1000'] > 5:
            risk_score += 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['certainty_per_1000'] > 3:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Плохое соотношение hedging/certainty
        if results['hedging_ratio'] < 0.5 and results['certainty_count'] > 5:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Личные местоимения (слишком мало - безличный стиль, может быть ИИ)
        if results['personal_per_1000'] < 0.5 and results['total_words'] > 500:
            risk_score += 1
            confidence = min(confidence + 0.1, 1.0)
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 5:
            results['risk_level'] = 'high'
        elif risk_score >= 3:
            results['risk_level'] = 'medium-high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        return results


class ParenthesisAnalyzer:
    """Анализ длинных пояснений в скобках (новый модуль)"""
    
    def __init__(self):
        self.min_words_for_long = 5  # 5+ слов в скобках - признак человека
    
    def analyze(self, text: str) -> Dict:
        """Анализирует содержимое круглых скобок"""
        results = {
            'total_parentheses': 0,
            'short_parentheses': 0,  # <5 слов
            'long_parentheses': 0,   # >=5 слов
            'long_percentage': 0,
            'long_examples': [],
            'words_in_parentheses': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'mean_words_in_parentheses': 0,
                'median_words_in_parentheses': 0,
                'max_words_in_parentheses': 0,
                'distribution': []
            }
        }
        
        if not text:
            return results
        
        # Находим все вхождения в круглых скобках
        pattern = r'\(([^)]+)\)'
        matches = re.findall(pattern, text)
        
        results['total_parentheses'] = len(matches)
        
        word_counts = []
        
        for match in matches[:100]:  # Ограничиваем для производительности
            words = match.split()
            word_count = len(words)
            word_counts.append(word_count)
            
            if word_count >= self.min_words_for_long:
                results['long_parentheses'] += 1
                if len(results['long_examples']) < 10:
                    results['long_examples'].append({
                        'text': match[:100] + '...' if len(match) > 100 else match,
                        'word_count': word_count
                    })
            else:
                results['short_parentheses'] += 1
            
            results['words_in_parentheses'].append(word_count)
        
        if results['total_parentheses'] > 0:
            results['long_percentage'] = (results['long_parentheses'] / results['total_parentheses']) * 100
        
        # Статистика
        if word_counts:
            results['statistics']['mean_words_in_parentheses'] = float(np.mean(word_counts))
            results['statistics']['median_words_in_parentheses'] = float(np.median(word_counts))
            results['statistics']['max_words_in_parentheses'] = float(np.max(word_counts))
            results['statistics']['distribution'] = word_counts
        
        # Оценка риска (чем больше длинных пояснений, тем ниже риск ИИ)
        risk_score = 0
        confidence = 0.5
        
        if results['long_percentage'] > 15:
            risk_score = 0  # Низкий риск (человек)
            confidence = 0.8
        elif results['long_percentage'] > 8:
            risk_score = 1  # Средне-низкий
            confidence = 0.6
        elif results['long_percentage'] > 3:
            risk_score = 2  # Средний
            confidence = 0.5
        else:
            risk_score = 3  # Высокий риск (ИИ не делает длинных пояснений)
            confidence = 0.7
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        else:
            results['risk_level'] = 'very_low'
        
        return results


class PunctuationAnalyzer:
    """Анализ пунктуации ! ? ; (новый модуль)"""
    
    def __init__(self):
        self.punctuation_marks = {
            'exclamation': '!',
            'question': '?',
            'semicolon': ';'
        }
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует использование редких знаков пунктуации"""
        results = {
            'exclamation_count': 0,
            'question_count': 0,
            'semicolon_count': 0,
            'exclamation_per_1000': 0,
            'question_per_1000': 0,
            'semicolon_per_1000': 0,
            'semicolon_contexts': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'mean_punctuation_per_sentence': 0,
                'median_punctuation_per_sentence': 0,
                'max_punctuation_in_sentence': 0,
                'distribution': {}
            }
        }
        
        if not text or not sentences:
            return results
        
        words = text.split()
        total_words = len(words)
        
        # Подсчет знаков пунктуации
        results['exclamation_count'] = text.count('!')
        results['question_count'] = text.count('?')
        results['semicolon_count'] = text.count(';')
        
        # Нормировка на 1000 слов
        if total_words > 0:
            results['exclamation_per_1000'] = (results['exclamation_count'] * 1000) / total_words
            results['question_per_1000'] = (results['question_count'] * 1000) / total_words
            results['semicolon_per_1000'] = (results['semicolon_count'] * 1000) / total_words
        
        # Контекст для точки с запятой (особенно важна)
        semicolon_positions = [m.start() for m in re.finditer(r';', text)]
        for pos in semicolon_positions[:10]:
            context = text[max(0, pos-50):min(len(text), pos+50)]
            # Проверяем, является ли это разделением длинного и короткого предложения
            before = context[:50].split(';')[0] if ';' in context[:50] else ''
            after = context[50:].split(';')[0] if ';' in context[50:] else ''
            
            before_words = len(before.split())
            after_words = len(after.split())
            
            results['semicolon_contexts'].append({
                'context': context.replace('\n', ' ').strip(),
                'before_words': before_words,
                'after_words': after_words,
                'is_contrast': after_words < before_words * 0.3  # Короткое после длинного
            })
        
        # Статистика по предложениям
        punct_per_sentence = []
        for sent in sentences[:100]:
            sent_punct = sent.count('!') + sent.count('?') + sent.count(';')
            punct_per_sentence.append(sent_punct)
        
        if punct_per_sentence:
            results['statistics']['mean_punctuation_per_sentence'] = float(np.mean(punct_per_sentence))
            results['statistics']['median_punctuation_per_sentence'] = float(np.median(punct_per_sentence))
            results['statistics']['max_punctuation_in_sentence'] = float(np.max(punct_per_sentence))
            results['statistics']['distribution'] = {
                'exclamation': results['exclamation_count'],
                'question': results['question_count'],
                'semicolon': results['semicolon_count']
            }
        
        # Оценка риска (редкие знаки - признак человека)
        risk_score = 0
        confidence = 0.5
        
        # Если есть восклицательные знаки (редко в науке) - признак человека
        if results['exclamation_per_1000'] > 0.1:
            risk_score -= 1  # Снижаем риск
            confidence = min(confidence + 0.1, 1.0)
        
        # Если есть вопросительные знаки (риторические вопросы) - признак человека
        if results['question_per_1000'] > 0.2:
            risk_score -= 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Точка с запятой - сильный признак человека (ИИ избегает)
        if results['semicolon_per_1000'] > 0.3:
            risk_score -= 2
            confidence = min(confidence + 0.2, 1.0)
        elif results['semicolon_per_1000'] > 0.1:
            risk_score -= 1
            confidence = min(confidence + 0.1, 1.0)
        
        # Нормализуем риск (не может быть отрицательным)
        risk_score = max(0, risk_score)
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score == 0:
            results['risk_level'] = 'very_low'
        elif risk_score == 1:
            results['risk_level'] = 'low'
        elif risk_score == 2:
            results['risk_level'] = 'medium'
        else:
            results['risk_level'] = 'high'
        
        return results


class ApostropheAnalyzer:
    """Анализ апострофов 's (новый модуль)"""
    
    def __init__(self):
        self.apostrophe_pattern = r"\w+'\w+"
    
    def analyze(self, text: str) -> Dict:
        """Анализирует использование апострофов"""
        results = {
            'apostrophe_count': 0,
            'apostrophe_per_1000': 0,
            'possessive_examples': [],
            'contraction_examples': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'mean_apostrophes_per_paragraph': 0,
                'median_apostrophes_per_paragraph': 0,
                'max_apostrophes_in_paragraph': 0,
                'distribution': []
            }
        }
        
        if not text:
            return results
        
        words = text.split()
        total_words = len(words)
        
        # Находим все слова с апострофами
        apostrophe_matches = re.findall(self.apostrophe_pattern, text)
        results['apostrophe_count'] = len(apostrophe_matches)
        
        # Нормировка на 1000 слов
        if total_words > 0:
            results['apostrophe_per_1000'] = (results['apostrophe_count'] * 1000) / total_words
        
        # Классифицируем примеры
        for match in apostrophe_matches[:20]:
            if match.endswith("'s") and len(match) > 2:
                # Возможно притяжательный падеж
                base_word = match[:-2]
                if base_word.isalpha():
                    results['possessive_examples'].append(match)
            elif any(cont in match for cont in ["'t", "'re", "'ve", "'ll", "'m", "'d"]):
                # Сокращение
                results['contraction_examples'].append(match)
        
        # Статистика по абзацам
        paragraphs = re.split(r'\n\s*\n', text)
        apostrophes_per_paragraph = []
        
        for para in paragraphs[:50]:
            para_apostrophes = len(re.findall(self.apostrophe_pattern, para))
            apostrophes_per_paragraph.append(para_apostrophes)
        
        if apostrophes_per_paragraph:
            results['statistics']['mean_apostrophes_per_paragraph'] = float(np.mean(apostrophes_per_paragraph))
            results['statistics']['median_apostrophes_per_paragraph'] = float(np.median(apostrophes_per_paragraph))
            results['statistics']['max_apostrophes_in_paragraph'] = float(np.max(apostrophes_per_paragraph))
            results['statistics']['distribution'] = apostrophes_per_paragraph
        
        # Оценка риска (частое употребление апострофов - признак ИИ)
        risk_score = 0
        confidence = 0.5
        
        if results['apostrophe_per_1000'] > 2.0:
            risk_score = 3
            confidence = 0.9
        elif results['apostrophe_per_1000'] > 1.0:
            risk_score = 2
            confidence = 0.7
        elif results['apostrophe_per_1000'] > 0.3:
            risk_score = 1
            confidence = 0.6
        else:
            risk_score = 0
            confidence = 0.5
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        else:
            results['risk_level'] = 'very_low'
        
        return results


class EnumerationAnalyzer:
    """Анализ строгих перечислений X, Y, and Z (новый модуль)"""
    
    def __init__(self):
        # Паттерн для трехэлементных перечислений: слово, слово, and слово
        self.three_item_pattern = r'\b(\w+),\s+(\w+),\s+and\s+(\w+)\b'
        # Паттерн для перечислений с specifically
        self.specifically_pattern = r'\bspecifically\s+(\w+),\s+(\w+),\s+and\s+(\w+)\b'
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует строгие перечисления из трех элементов"""
        results = {
            'three_item_count': 0,
            'three_item_per_1000_sentences': 0,
            'specifically_count': 0,
            'examples': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'statistics': {
                'mean_enumerations_per_sentence': 0,
                'max_enumerations_in_sentence': 0,
                'distribution': []
            }
        }
        
        if not text or not sentences:
            return results
        
        total_sentences = len(sentences)
        
        # Поиск трехэлементных перечислений
        three_item_matches = re.findall(self.three_item_pattern, text, re.IGNORECASE)
        results['three_item_count'] = len(three_item_matches)
        
        # Поиск перечислений с specifically
        specifically_matches = re.findall(self.specifically_pattern, text, re.IGNORECASE)
        results['specifically_count'] = len(specifically_matches)
        
        # Нормировка на 1000 предложений
        if total_sentences > 0:
            results['three_item_per_1000_sentences'] = (results['three_item_count'] * 1000) / total_sentences
        
        # Собираем примеры
        for match in three_item_matches[:10]:
            example = f"{match[0]}, {match[1]}, and {match[2]}"
            results['examples'].append(example)
        
        # Статистика по предложениям
        enumerations_per_sentence = []
        for sent in sentences[:100]:
            sent_enumerations = len(re.findall(self.three_item_pattern, sent, re.IGNORECASE))
            enumerations_per_sentence.append(sent_enumerations)
        
        if enumerations_per_sentence:
            results['statistics']['mean_enumerations_per_sentence'] = float(np.mean(enumerations_per_sentence))
            results['statistics']['max_enumerations_in_sentence'] = float(np.max(enumerations_per_sentence))
            results['statistics']['distribution'] = enumerations_per_sentence
        
        # Оценка риска (частое использование строгих перечислений - признак ИИ)
        risk_score = 0
        confidence = 0.5
        
        if results['three_item_per_1000_sentences'] > 10:
            risk_score = 3
            confidence = 0.9
        elif results['three_item_per_1000_sentences'] > 5:
            risk_score = 2
            confidence = 0.7
        elif results['three_item_per_1000_sentences'] > 2:
            risk_score = 1
            confidence = 0.6
        else:
            risk_score = 0
            confidence = 0.5
        
        # Перечисления с specifically - еще более сильный признак
        if results['specifically_count'] > 3:
            risk_score = min(risk_score + 1, 3)
            confidence = min(confidence + 0.1, 1.0)
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        else:
            results['risk_level'] = 'very_low'
        
        return results


class ParagraphAnalyzer:
    """Анализ на уровне абзацев (новый модуль)"""
    
    def __init__(self):
        self.available = SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        if self.available:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Легкая модель
            except:
                self.available = False
    
    def split_paragraphs(self, text: str) -> List[str]:
        """Разбивает текст на абзацы"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует однородность внутри и между абзацами"""
        results = {
            'paragraphs': [],
            'paragraph_lengths': [],
            'intra_paragraph_similarity': 0,
            'inter_paragraph_similarity': 0,
            'transition_smoothness': 0,
            'structure_repetition': 0,
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'note': 'Paragraph analysis requires sentence-transformers',
            'statistics': {
                'mean_paragraph_length': 0,
                'median_paragraph_length': 0,
                'max_paragraph_length': 0,
                'min_paragraph_length': 0,
                'std_paragraph_length': 0,
                'distribution': []
            }
        }
        
        if not self.available or len(text) < 500:
            return results
        
        paragraphs = self.split_paragraphs(text)
        if len(paragraphs) < 3:
            return results
        
        results['paragraphs'] = paragraphs[:15]  # Ограничиваем
        
        # Статистика длины абзацев
        para_lengths = [len(p.split()) for p in paragraphs]
        results['paragraph_lengths'] = para_lengths[:50]
        
        if para_lengths:
            results['statistics']['mean_paragraph_length'] = float(np.mean(para_lengths))
            results['statistics']['median_paragraph_length'] = float(np.median(para_lengths))
            results['statistics']['max_paragraph_length'] = float(np.max(para_lengths))
            results['statistics']['min_paragraph_length'] = float(np.min(para_lengths))
            results['statistics']['std_paragraph_length'] = float(np.std(para_lengths))
            results['statistics']['distribution'] = para_lengths[:100]
        
        # Если нет модели, используем простые метрики
        if not self.model:
            # Простая метрика: вариативность длины абзацев
            if len(para_lengths) > 1:
                cv_lengths = np.std(para_lengths) / np.mean(para_lengths) if np.mean(para_lengths) > 0 else 0
                if cv_lengths < 0.3:
                    results['risk_score'] = 2
                    results['confidence'] = 0.6
                    results['risk_level'] = 'medium'
                    results['note'] = 'Simple paragraph length analysis'
            return results
        
        # Полноценный анализ с эмбеддингами
        try:
            # Получаем эмбеддинги для абзацев
            para_embeddings = self.model.encode(paragraphs[:25])  # Ограничиваем
            
            # Внутриабзацная похожесть (разбиваем каждый абзац на предложения)
            intra_similarities = []
            for para in paragraphs[:8]:  # Первые 8 абзацев
                para_sents = [s for s in re.split(r'[.!?]+', para) if len(s.split()) > 5]
                if len(para_sents) >= 3:
                    sent_embeddings = self.model.encode(para_sents[:10])
                    # Средняя попарная косинусная близость
                    similarities = []
                    for i in range(len(sent_embeddings)):
                        for j in range(i+1, len(sent_embeddings)):
                            sim = np.dot(sent_embeddings[i], sent_embeddings[j]) / (
                                np.linalg.norm(sent_embeddings[i]) * np.linalg.norm(sent_embeddings[j]))
                            similarities.append(sim)
                    if similarities:
                        intra_similarities.append(np.mean(similarities))
            
            if intra_similarities:
                results['intra_paragraph_similarity'] = float(np.mean(intra_similarities))
            
            # Межабзацная похожесть
            if len(para_embeddings) > 1:
                inter_similarities = []
                for i in range(len(para_embeddings) - 1):
                    sim = np.dot(para_embeddings[i], para_embeddings[i+1]) / (
                        np.linalg.norm(para_embeddings[i]) * np.linalg.norm(para_embeddings[i+1]))
                    inter_similarities.append(sim)
                results['inter_paragraph_similarity'] = float(np.mean(inter_similarities))
            
            # Плавность переходов (последнее предложение абзаца - первое следующего)
            transition_sims = []
            for i in range(len(paragraphs) - 1):
                curr_sents = [s for s in re.split(r'[.!?]+', paragraphs[i]) if len(s.split()) > 5]
                next_sents = [s for s in re.split(r'[.!?]+', paragraphs[i+1]) if len(s.split()) > 5]
                
                if curr_sents and next_sents:
                    last_curr = self.model.encode([curr_sents[-1]])[0]
                    first_next = self.model.encode([next_sents[0]])[0]
                    sim = np.dot(last_curr, first_next) / (np.linalg.norm(last_curr) * np.linalg.norm(first_next))
                    transition_sims.append(sim)
            
            if transition_sims:
                results['transition_smoothness'] = float(np.mean(transition_sims))
            
            # Оценка риска
            risk_score = 0
            confidence = 0.5
            
            # Высокая внутриабзацная похожесть (>0.72) - подозрительно
            if results['intra_paragraph_similarity'] > 0.72:
                risk_score += 3
                confidence = min(confidence + 0.3, 1.0)
            elif results['intra_paragraph_similarity'] > 0.65:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif results['intra_paragraph_similarity'] > 0.6:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
            
            # Очень плавные переходы (>0.65) - подозрительно
            if results['transition_smoothness'] > 0.65:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif results['transition_smoothness'] > 0.55:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
            
            # Однородность абзацев
            if results['inter_paragraph_similarity'] > 0.5:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
            
            results['risk_score'] = risk_score
            results['confidence'] = confidence
            
            if risk_score >= 4:
                results['risk_level'] = 'high'
            elif risk_score >= 2:
                results['risk_level'] = 'medium'
            elif risk_score >= 1:
                results['risk_level'] = 'low'
            
            results['note'] = 'Full semantic paragraph analysis'
            
        except Exception as e:
            results['note'] = f'Error in paragraph analysis: {str(e)}'
        
        return results


class PerplexityAnalyzer:
    """Анализ перплексии (полная версия)"""
    
    def __init__(self):
        self.available = TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
        self.perplexity_pipeline = None
        
        if self.available:
            try:
                # Используем небольшую модель для перплексии
                model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.perplexity_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=512
                )
            except Exception as e:
                self.available = False
                print(f"Error loading perplexity model: {e}")
    
    def calculate_perplexity(self, text: str) -> float:
        """Вычисляет перплексию текста"""
        if not self.available or not text:
            return 0
        
        try:
            # Токенизируем
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Вычисляем loss
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            return 0
    
    def analyze(self, text: str) -> Dict:
        """Полный анализ перплексии"""
        results = {
            'perplexities': [],
            'mean_perplexity': 0,
            'median_perplexity': 0,
            'perplexity_variance': 0,
            'min_perplexity': 0,
            'max_perplexity': 0,
            'perplexity_segments': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'note': 'Perplexity analysis requires transformers library',
            'statistics': {
                'distribution': []
            }
        }
        
        if not self.available or len(text) < 100:
            results['note'] = 'Perplexity analysis unavailable or text too short'
            return results
        
        try:
            # Разбиваем текст на сегменты по 500 токенов
            words = text.split()
            segment_size = 500
            segments = [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]
            segments = segments[:20]  # Ограничиваем для производительности
            
            perplexities = []
            
            for i, segment in enumerate(segments):
                if len(segment.split()) < 50:
                    continue
                
                ppl = self.calculate_perplexity(segment)
                if ppl > 0:
                    perplexities.append(ppl)
                    results['perplexities'].append(ppl)
                    results['perplexity_segments'].append({
                        'segment': i,
                        'perplexity': ppl
                    })
            
            if perplexities:
                results['mean_perplexity'] = float(np.mean(perplexities))
                results['median_perplexity'] = float(np.median(perplexities))
                results['min_perplexity'] = float(np.min(perplexities))
                results['max_perplexity'] = float(np.max(perplexities))
                results['perplexity_variance'] = float(np.var(perplexities))
                results['statistics']['distribution'] = perplexities
            
            # Оценка риска
            risk_score = 0
            confidence = 0.5
            
            if results['mean_perplexity'] < 15:
                risk_score = 3  # Низкая перплексия - вероятно ИИ
                confidence = 0.8
            elif results['mean_perplexity'] < 25:
                risk_score = 2
                confidence = 0.7
            elif results['mean_perplexity'] < 40:
                risk_score = 1
                confidence = 0.6
            else:
                risk_score = 0  # Высокая перплексия - человек
                confidence = 0.5
            
            results['risk_score'] = risk_score
            results['confidence'] = confidence
            
            if risk_score >= 3:
                results['risk_level'] = 'high'
            elif risk_score >= 2:
                results['risk_level'] = 'medium'
            elif risk_score >= 1:
                results['risk_level'] = 'low'
            else:
                results['risk_level'] = 'very_low'
            
            results['note'] = 'Full perplexity analysis'
            
        except Exception as e:
            results['note'] = f'Error in perplexity analysis: {str(e)}'
        
        return results


class SemanticAnalyzer:
    """Анализ семантической близости (полная версия)"""
    
    def __init__(self):
        self.available = SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        if self.available:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.available = False
    
    def analyze(self, sentences: List[str]) -> Dict:
        """Полный семантический анализ"""
        results = {
            'similarities': [],
            'mean_similarity': 0,
            'similarity_variance': 0,
            'min_similarity': 0,
            'max_similarity': 0,
            'similarity_matrix': [],
            'semantic_clusters': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'note': 'Semantic analysis requires sentence-transformers',
            'statistics': {
                'distribution': []
            }
        }
        
        if not self.available or len(sentences) < 10:
            return results
        
        try:
            # Берем первые 50 предложений для анализа
            sample_sentences = [s for s in sentences if len(s.split()) > 5][:50]
            
            if len(sample_sentences) < 5:
                return results
            
            # Получаем эмбеддинги
            embeddings = self.model.encode(sample_sentences)
            
            # Вычисляем попарные косинусные сходства
            similarities = []
            n = len(embeddings)
            
            for i in range(n):
                for j in range(i+1, n):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    similarities.append(sim)
            
            if similarities:
                results['similarities'] = similarities[:1000]  # Ограничиваем
                results['mean_similarity'] = float(np.mean(similarities))
                results['similarity_variance'] = float(np.var(similarities))
                results['min_similarity'] = float(np.min(similarities))
                results['max_similarity'] = float(np.max(similarities))
                results['statistics']['distribution'] = similarities[:500]
                
                # Простая кластеризация (по порогу)
                threshold = 0.7
                clusters = 0
                used = set()
                
                for i in range(n):
                    if i in used:
                        continue
                    cluster = [i]
                    for j in range(i+1, n):
                        if j in used:
                            continue
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                        if sim > threshold:
                            cluster.append(j)
                            used.add(j)
                    if len(cluster) > 1:
                        clusters += 1
                    used.add(i)
                
                results['semantic_clusters'] = clusters
            
            # Оценка риска
            risk_score = 0
            confidence = 0.5
            
            if results['mean_similarity'] > 0.65:
                risk_score = 3  # Высокая семантическая близость - ИИ
                confidence = 0.8
            elif results['mean_similarity'] > 0.55:
                risk_score = 2
                confidence = 0.7
            elif results['mean_similarity'] > 0.45:
                risk_score = 1
                confidence = 0.6
            else:
                risk_score = 0
                confidence = 0.5
            
            results['risk_score'] = risk_score
            results['confidence'] = confidence
            
            if risk_score >= 3:
                results['risk_level'] = 'high'
            elif risk_score >= 2:
                results['risk_level'] = 'medium'
            elif risk_score >= 1:
                results['risk_level'] = 'low'
            else:
                results['risk_level'] = 'very_low'
            
            results['note'] = 'Full semantic analysis'
            
        except Exception as e:
            results['note'] = f'Error in semantic analysis: {str(e)}'
        
        return results


class IntegratedRiskScorer:
    """Интегральная оценка риска на основе всех модулей"""
    
    def __init__(self):
        # Веса модулей (обновленные с учетом новых модулей)
        self.weights = {
            'unicode': 0.05,      # 5%
            'dashes': 0.05,        # 5%
            'phrases': 0.10,       # 10%
            'burstiness': 0.08,    # 8%
            'grammar': 0.10,       # 10%
            'hedging': 0.15,       # 15% (ключевой)
            'paragraph': 0.10,     # 10%
            'perplexity': 0.08,    # 8%
            'semantic': 0.08,      # 8%
            'parenthesis': 0.06,   # 6% (новый)
            'punctuation': 0.05,   # 5% (новый)
            'apostrophe': 0.05,    # 5% (новый)
            'enumeration': 0.05    # 5% (новый)
        }
        
        # Нормируем веса
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
    
    def calculate(self, results: Dict) -> Dict:
        """Вычисляет интегральный риск"""
        total_score = 0
        total_confidence = 0
        weighted_score = 0
        module_scores = []
        
        for module, weight in self.weights.items():
            if module in results and results[module]:
                data = results[module]
                if 'risk_score' in data and data['risk_score'] is not None:
                    # Нормируем risk_score (0-3) к 0-1
                    max_score = 3
                    norm_score = min(data['risk_score'] / max_score, 1.0)
                    
                    # Учитываем уверенность модуля
                    confidence = data.get('confidence', 0.5)
                    
                    # Для модулей, где низкий риск - признак человека, инвертируем
                    invert_modules = ['parenthesis', 'punctuation']
                    if module in invert_modules and 'risk_score' in data:
                        # Эти модули: высокий score = низкий риск ИИ
                        norm_score = 1.0 - norm_score
                    
                    module_score = {
                        'module': module,
                        'raw_score': data['risk_score'],
                        'norm_score': norm_score,
                        'weight': weight,
                        'confidence': confidence,
                        'contribution': norm_score * weight * 100
                    }
                    
                    module_scores.append(module_score)
                    weighted_score += norm_score * weight
                    total_confidence += confidence * weight
        
        # Итоговая оценка 0-100
        final_score = weighted_score * 100
        
        # Корректировка на уверенность
        if total_confidence > 0:
            final_score = final_score * (0.5 + 0.5 * total_confidence)
        
        # Определяем уровень риска
        risk_level = 'unknown'
        if final_score < 20:
            risk_level = 'very_low'
        elif final_score < 35:
            risk_level = 'low'
        elif final_score < 50:
            risk_level = 'medium-low'
        elif final_score < 65:
            risk_level = 'medium'
        elif final_score < 80:
            risk_level = 'medium-high'
        else:
            risk_level = 'high'
        
        return {
            'final_score': final_score,
            'risk_level': risk_level,
            'weighted_score': weighted_score,
            'total_confidence': total_confidence,
            'module_scores': module_scores
        }


class DocumentProcessor:
    """Обработчик загруженных документов"""
    
    @staticmethod
    def read_docx(file) -> Optional[str]:
        """Читает .docx файл"""
        try:
            from docx import Document
            doc = Document(file)
            full_text = []
            for para in doc.paragraphs:
                if para.text:
                    full_text.append(para.text)
            
            # Читаем таблицы
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            if para.text:
                                full_text.append(para.text)
            
            return '\n'.join(full_text)
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return None
    
    @staticmethod
    def read_doc(file) -> Optional[str]:
        """Читает .doc файл (если доступен antiword)"""
        try:
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc', mode='wb') as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            # Пробуем использовать antiword
            try:
                result = subprocess.run(['antiword', tmp_path], 
                                       capture_output=True, text=True, timeout=10)
                os.unlink(tmp_path)
                if result.returncode == 0 and result.stdout:
                    return result.stdout
            except:
                pass
            
            # Если antiword не сработал, пробуем текстовый fallback
            os.unlink(tmp_path)
            
            # Пробуем прочитать как текстовый файл
            file.seek(0)
            content = file.getvalue().decode('utf-8', errors='ignore')
            if len(content) > 100:  # Хотя бы что-то получили
                return content
            
            return None
        except Exception as e:
            return None
    
    @staticmethod
    def split_sentences_simple(text: str) -> List[str]:
        """Простая сегментация на предложения без spacy"""
        # Регулярное выражение для разделения на предложения
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    @staticmethod
    def preprocess(text: str) -> str:
        """Базовая предобработка текста"""
        if not text:
            return ""
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
    
    # Левая боковая панель с загрузкой и настройками
    with st.sidebar:
        st.markdown("## 📁 Загрузка файлов")
        
        uploaded_file = st.file_uploader(
            "Выберите файл статьи", 
            type=['docx', 'doc'],
            help="Поддерживаются форматы .docx и .doc"
        )
        
        if uploaded_file is not None:
            st.markdown("### 📄 Информация о файле")
            file_details = {
                "Имя": uploaded_file.name,
                "Размер": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
        
        st.markdown("---")
        st.markdown("### ⚙️ Настройки анализа")
        
        # Опции анализа
        cut_references = st.checkbox("Обрезать до References", value=True, 
                                     help="Не анализировать список литературы")
        deep_analysis = st.checkbox("Глубокий анализ (медленнее)", value=False,
                                    help="Включает полный семантический анализ и перплексию")
        
        st.markdown("---")
        st.markdown("### 🎯 Активные модули")
        st.markdown("""
        - ✅ Unicode-артефакты
        - ✅ Множественные тире
        - ✅ ИИ-фразы 2025-2026
        - ✅ Burstiness
        - ✅ Грамматика
        - ✅ Хеджинг (ключевой)
        - ✅ Длинные скобки (>5 слов)
        - ✅ Пунктуация (! ? ;)
        - ✅ Апострофы 's
        - ✅ Перечисления X, Y, and Z
        - ✅ Абзацный анализ
        - ✅ Перплексия (если доступно)
        - ✅ Семантика (если доступно)
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Интегральная оценка")
        st.markdown("**Веса модулей 2025-2026:**")
        st.markdown("- Хеджинг: 15%")
        st.markdown("- ИИ-фразы: 10%")
        st.markdown("- Грамматика: 10%")
        st.markdown("- Абзацы: 10%")
        st.markdown("- Перплексия: 8%")
        st.markdown("- Семантика: 8%")
        st.markdown("- Burstiness: 8%")
        st.markdown("- Скобки: 6% (новое)")
        st.markdown("- Пунктуация: 5% (новое)")
        st.markdown("- Апострофы: 5% (новое)")
        st.markdown("- Перечисления: 5% (новое)")
        st.markdown("- Остальное: 10%")
        
        if st.button("🔄 Начать анализ", use_container_width=True):
            st.session_state['analyze'] = True
    
    # Основная область - три вкладки
    if uploaded_file is not None and st.session_state.get('analyze', False):
        # Прогресс
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Шаг 1: Чтение документа
            status_text.text("Чтение документа...")
            progress_bar.progress(5)
            
            if uploaded_file.name.endswith('.docx'):
                text = DocumentProcessor.read_docx(uploaded_file)
            else:  # .doc
                text = DocumentProcessor.read_doc(uploaded_file)
            
            if not text or len(text.strip()) < 100:
                st.warning("Файл слишком мал или не содержит текста")
                return
            
            # Шаг 2: Обрезка до References (если выбрано)
            if cut_references:
                status_text.text("Обрезка до раздела References...")
                text = ReferenceCutoff.cut_at_references(text)
            
            # Шаг 3: Предобработка
            status_text.text("Предобработка текста...")
            progress_bar.progress(10)
            text = DocumentProcessor.preprocess(text)
            
            # Шаг 4: Сегментация на предложения
            status_text.text("Сегментация текста...")
            progress_bar.progress(15)
            sentences = DocumentProcessor.split_sentences_simple(text)
            
            st.success(f"✅ Текст загружен: {len(text)} символов, {len(sentences)} предложений")
            
            # Создаем три основные вкладки
            tab_overview, tab_details, tab_text = st.tabs([
                "📊 Обзор", 
                "🔍 Детальный анализ", 
                "📝 Текст с разметкой"
            ])
            
            # Хранилище результатов
            results = {}
            risk_scores = []
            
            # =================================================================
            # Модуль 1: Unicode-артефакты
            # =================================================================
            with st.spinner("Анализ Unicode-артефактов и нелатинских символов..."):
                detector = UnicodeArtifactDetector()
                results['unicode'] = detector.analyze(text)
                risk_scores.append(results['unicode']['risk_score'])
            progress_bar.progress(12)
            
            # =================================================================
            # Модуль 2: Множественные тире
            # =================================================================
            with st.spinner("Анализ множественных тире..."):
                detector = DashAnalyzer()
                results['dashes'] = detector.analyze(sentences)
                risk_scores.append(results['dashes']['risk_score'])
            progress_bar.progress(18)
            
            # =================================================================
            # Модуль 3: ИИ-фразы
            # =================================================================
            with st.spinner("Поиск ИИ-фраз (обновленный список 2025-2026)..."):
                detector = AIPhraseDetector()
                results['phrases'] = detector.analyze(text, sentences)
                risk_scores.append(results['phrases']['risk_score'])
            progress_bar.progress(24)
            
            # =================================================================
            # Модуль 4: Burstiness
            # =================================================================
            with st.spinner("Анализ вариативности предложений..."):
                detector = BurstinessAnalyzer()
                results['burstiness'] = detector.analyze(sentences)
                if 'error' not in results['burstiness']:
                    risk_scores.append(results['burstiness']['risk_score'])
            progress_bar.progress(30)
            
            # =================================================================
            # Модуль 5: Грамматика
            # =================================================================
            with st.spinner("Грамматический анализ (модальность, номинализации)..."):
                detector = GrammarAnalyzer()
                results['grammar'] = detector.analyze(text)
                risk_scores.append(results['grammar']['risk_score'])
            progress_bar.progress(36)
            
            # =================================================================
            # Модуль 6: Хеджинг
            # =================================================================
            with st.spinner("Анализ хеджинга (ключевой маркер)..."):
                detector = HedgingAnalyzer()
                results['hedging'] = detector.analyze(text)
                risk_scores.append(results['hedging']['risk_score'])
            progress_bar.progress(42)
            
            # =================================================================
            # Модуль 7: Длинные скобки (новый)
            # =================================================================
            with st.spinner("Анализ длинных пояснений в скобках..."):
                detector = ParenthesisAnalyzer()
                results['parenthesis'] = detector.analyze(text)
                risk_scores.append(results['parenthesis']['risk_score'])
            progress_bar.progress(48)
            
            # =================================================================
            # Модуль 8: Пунктуация (новый)
            # =================================================================
            with st.spinner("Анализ пунктуации ! ? ; ..."):
                detector = PunctuationAnalyzer()
                results['punctuation'] = detector.analyze(text, sentences)
                risk_scores.append(results['punctuation']['risk_score'])
            progress_bar.progress(54)
            
            # =================================================================
            # Модуль 9: Апострофы (новый)
            # =================================================================
            with st.spinner("Анализ апострофов 's ..."):
                detector = ApostropheAnalyzer()
                results['apostrophe'] = detector.analyze(text)
                risk_scores.append(results['apostrophe']['risk_score'])
            progress_bar.progress(60)
            
            # =================================================================
            # Модуль 10: Перечисления (новый)
            # =================================================================
            with st.spinner("Анализ перечислений X, Y, and Z ..."):
                detector = EnumerationAnalyzer()
                results['enumeration'] = detector.analyze(text, sentences)
                risk_scores.append(results['enumeration']['risk_score'])
            progress_bar.progress(66)
            
            # =================================================================
            # Модуль 11: Абзацный анализ
            # =================================================================
            with st.spinner("Анализ на уровне абзацев..."):
                detector = ParagraphAnalyzer()
                results['paragraph'] = detector.analyze(text, sentences)
                if 'error' not in results['paragraph']:
                    risk_scores.append(results['paragraph']['risk_score'])
            progress_bar.progress(75)
            
            # =================================================================
            # Модуль 12: Perplexity (если глубокий анализ)
            # =================================================================
            if deep_analysis:
                with st.spinner("Анализ перплексии..."):
                    detector = PerplexityAnalyzer()
                    results['perplexity'] = detector.analyze(text)
                    if 'error' not in results['perplexity']:
                        risk_scores.append(results['perplexity']['risk_score'])
            progress_bar.progress(85)
            
            # =================================================================
            # Модуль 13: Семантика (если глубокий анализ)
            # =================================================================
            if deep_analysis:
                with st.spinner("Семантический анализ..."):
                    detector = SemanticAnalyzer()
                    results['semantic'] = detector.analyze(sentences)
                    if 'error' not in results['semantic']:
                        risk_scores.append(results['semantic']['risk_score'])
            progress_bar.progress(95)
            
            # Интегральная оценка
            scorer = IntegratedRiskScorer()
            integrated_results = scorer.calculate(results)
            
            # Завершение
            progress_bar.progress(100)
            status_text.text("✅ Анализ завершен!")
            
            # =================================================================
            # Вкладка 1: Обзор (ключевые метрики)
            # =================================================================
            with tab_overview:
                st.markdown("## 📊 Общая оценка")
                
                # Интегральный риск в виде крупной метрики
                final_score = integrated_results['final_score']
                risk_level = integrated_results['risk_level']
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if risk_level == 'very_low':
                        st.markdown(f"""
                        <div class="risk-low">
                            <h2 style="color: #2D6A4F;">🟢 Очень низкий риск</h2>
                            <h1>{final_score:.1f}/100</h1>
                            <p>Текст демонстрирует сильные признаки человеческого письма:
                            длинные пояснения в скобках, разнообразие пунктуации, естественные переходы.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == 'low':
                        st.markdown(f"""
                        <div class="risk-low">
                            <h2 style="color: #2D6A4F;">🟢 Низкий риск</h2>
                            <h1>{final_score:.1f}/100</h1>
                            <p>Текст преимущественно похож на человеческий, есть некоторые признаки ИИ.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == 'medium-low':
                        st.markdown(f"""
                        <div class="risk-medium">
                            <h2 style="color: #B45309;">🟡 Умеренно-низкий риск</h2>
                            <h1>{final_score:.1f}/100</h1>
                            <p>Текст имеет смешанные признаки. Рекомендуется дополнительная проверка.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == 'medium':
                        st.markdown(f"""
                        <div class="risk-medium">
                            <h2 style="color: #B45309;">🟠 Средний риск</h2>
                            <h1>{final_score:.1f}/100</h1>
                            <p>Текст демонстрирует множественные признаки ИИ-генерации.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level == 'medium-high':
                        st.markdown(f"""
                        <div class="risk-medium">
                            <h2 style="color: #B45309;">🟠 Умеренно-высокий риск</h2>
                            <h1>{final_score:.1f}/100</h1>
                            <p>Сильные признаки ИИ: низкий хеджинг, высокая однородность.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-high">
                            <h2 style="color: #991B1B;">🔴 Высокий риск</h2>
                            <h1>{final_score:.1f}/100</h1>
                            <p>Текст с высокой вероятностью сгенерирован ИИ. Обнаружены сильные маркеры.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Активных модулей", len(risk_scores))
                    st.metric("Уверенность", f"{integrated_results['total_confidence']:.0%}")
                
                with col3:
                    st.metric("Предложений", len(sentences))
                    st.metric("Символов", len(text))
                
                # Ключевые метрики в виде карточек
                st.markdown("### 🎯 Ключевые метрики")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    if 'hedging' in results:
                        h = results['hedging']
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Хеджинг</h4>
                            <h2>{h['hedging_per_1000']:.1f}/1000</h2>
                            <p>Уверенность: {h.get('confidence', 0.5):.0%}</p>
                            <small>{'⚠️ Низкий' if h['hedging_per_1000'] < 3 else '✓ Норма'}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_b:
                    if 'parenthesis' in results:
                        p = results['parenthesis']
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Длинные скобки</h4>
                            <h2>{p['long_percentage']:.1f}%</h2>
                            <p>Всего скобок: {p['total_parentheses']}</p>
                            <small>{'✓ Признак человека' if p['long_percentage'] > 8 else '⚠️ Мало'}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_c:
                    if 'punctuation' in results:
                        pu = results['punctuation']
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Пунктуация !?;</h4>
                            <h2>{pu['semicolon_per_1000']:.2f}/1000</h2>
                            <p>!:{pu['exclamation_per_1000']:.2f} ?:{pu['question_per_1000']:.2f}</p>
                            <small>{'✓ Разнообразие' if pu['semicolon_per_1000'] > 0.1 else '⚠️ Однообразно'}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_d:
                    if 'enumeration' in results:
                        e = results['enumeration']
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Перечисления</h4>
                            <h2>{e['three_item_per_1000_sentences']:.1f}/1000</h2>
                            <p>Всего: {e['three_item_count']}</p>
                            <small>{'⚠️ Часто (ИИ)' if e['three_item_per_1000_sentences'] > 5 else '✓ Норма'}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Диаграмма вклада модулей
                st.markdown("### 📊 Вклад модулей в итоговую оценку")
                
                if integrated_results['module_scores']:
                    module_df = pd.DataFrame(integrated_results['module_scores'])
                    module_df = module_df.sort_values('contribution', ascending=True)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=module_df['contribution'],
                            y=module_df['module'],
                            orientation='h',
                            marker=dict(
                                color=module_df['contribution'],
                                colorscale='RdYlGn_r',
                                showscale=True,
                                colorbar=dict(title="Вклад (%)")
                            ),
                            text=module_df['contribution'].round(1),
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="",
                        xaxis_title="Вклад в итоговую оценку (%)",
                        yaxis_title="",
                        height=500,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # =================================================================
            # Вкладка 2: Детальный анализ
            # =================================================================
            with tab_details:
                st.markdown("## 🔍 Детальный анализ по модулям")
                
                # Создаем вкладки для групп модулей
                detail_tabs = st.tabs([
                    "Артефакты", 
                    "Статистика",
                    "Грамматика",
                    "Новые модули",
                    "Семантика"
                ])
                
                # Вкладка "Артефакты"
                with detail_tabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'unicode' in results:
                            st.subheader("Unicode-артефакты")
                            u = results['unicode']
                            
                            st.metric("Плотность на 10k", f"{u['density_per_10k']:.2f}")
                            st.metric("Надстрочные/подстрочные", u['sup_sub_count'])
                            st.metric("Fullwidth цифры", u['fullwidth_count'])
                            st.metric("Гомоглифы", u['homoglyph_count'])
                            
                            # Статистика распределения
                            if u['statistics']['distribution']:
                                fig = go.Figure(data=[
                                    go.Histogram(x=u['statistics']['distribution'], nbinsx=20)
                                ])
                                fig.update_layout(
                                    title="Распределение плотности артефактов",
                                    xaxis_title="Плотность на 10k",
                                    yaxis_title="Частота",
                                    height=250
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            if u['suspicious_chunks']:
                                with st.expander("Примеры подозрительных символов"):
                                    for chunk in u['suspicious_chunks'][:5]:
                                        st.code(f"Символ '{chunk['char']}' в контексте: ...{chunk['context']}...")
                    
                    with col2:
                        if 'dashes' in results:
                            st.subheader("Множественные тире")
                            d = results['dashes']
                            
                            st.metric("Предложений с ≥2 тире", len(d['sentences_with_multiple_dashes']))
                            st.metric("Тяжелые предложения", len(d['heavy_sentences']))
                            st.metric("Процент тяжелых", f"{d['percentage_heavy']:.2f}%")
                            
                            # Статистика
                            if d['statistics']['distribution']:
                                fig = go.Figure(data=[
                                    go.Histogram(x=d['statistics']['distribution'], nbinsx=10)
                                ])
                                fig.update_layout(
                                    title="Распределение тире по предложениям",
                                    xaxis_title="Количество тире",
                                    yaxis_title="Частота",
                                    height=250
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            if d['examples']:
                                with st.expander("Примеры предложений с тире"):
                                    for ex in d['examples'][:3]:
                                        st.info(ex)
                
                # Вкладка "Статистика"
                with detail_tabs[1]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'burstiness' in results:
                            st.subheader("Burstiness (вариативность)")
                            b = results['burstiness']
                            
                            st.metric("Ср. длина предложения", f"{b['mean_length']:.1f} слов")
                            st.metric("Медианная длина", f"{b['statistics']['median_length']:.1f}")
                            st.metric("Макс. длина", f"{b['statistics']['max_length']:.0f}")
                            st.metric("Мин. длина", f"{b['statistics']['min_length']:.0f}")
                            st.metric("Коэф. вариации", f"{b['cv']:.3f}")
                            
                            # Гистограмма
                            if b['sentence_lengths']:
                                fig = go.Figure(data=[
                                    go.Histogram(x=b['sentence_lengths'][:200], nbinsx=30)
                                ])
                                fig.update_layout(
                                    title="Распределение длины предложений",
                                    xaxis_title="Длина (слова)",
                                    yaxis_title="Частота",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'phrases' in results:
                            st.subheader("ИИ-фразы")
                            p = results['phrases']
                            
                            st.metric("Плотность переходов", f"{p['transition_density']:.2f}/1000")
                            st.metric("Метадискурс на 1000", f"{p.get('metadiscourse_per_1000', 0):.2f}")
                            st.metric("Повторяющихся фраз", len(p['repeated_phrases']))
                            
                            # Статистика
                            st.metric("Ср. фраз на предложение", 
                                     f"{p['statistics']['mean_phrases_per_sentence']:.2f}")
                            
                            if p['top_phrases']:
                                with st.expander("Наиболее частые фразы"):
                                    for phrase, count in p['top_phrases'][:10]:
                                        st.write(f"- '{phrase}': {count} раз(а)")
                            
                            if p['repeated_phrases']:
                                with st.expander("⚠️ Часто повторяющиеся"):
                                    for item in p['repeated_phrases']:
                                        st.write(f"- '{item['phrase']}' ({item['count']} раз)")
                
                # Вкладка "Грамматика"
                with detail_tabs[2]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'grammar' in results:
                            st.subheader("Грамматика")
                            g = results['grammar']
                            
                            st.metric("Пассивных конструкций", f"{g['passive_percentage']:.1f}%")
                            st.metric("Модальные глаголы на 1000", f"{g.get('modals_per_1000', 0):.2f}")
                            st.metric("Номинализаций на 1000", f"{g['nominalizations_per_1000']:.1f}")
                            st.metric("Epistemic markers", f"{g.get('epistemic_per_1000', 0):.2f}")
                            st.metric("Усилители уверенности", f"{g.get('boosters_per_1000', 0):.2f}")
                            
                            # Статистика
                            st.metric("Ср. пассивных на предложение", 
                                     f"{g['statistics']['mean_passive_per_sentence']:.2f}")
                            
                            if g['examples']:
                                with st.expander("Примеры пассива"):
                                    for ex in g['examples'][:3]:
                                        st.info(ex)
                    
                    with col2:
                        if 'hedging' in results:
                            st.subheader("Хеджинг (ключевой маркер)")
                            h = results['hedging']
                            
                            st.metric("Хеджинг на 1000 слов", f"{h['hedging_per_1000']:.2f}")
                            st.metric("Категоричность на 1000", f"{h.get('certainty_per_1000', 0):.2f}")
                            st.metric("Личные местоимения на 1000", f"{h['personal_per_1000']:.2f}")
                            st.metric("Соотношение H/C", f"{h.get('hedging_ratio', 0):.2f}")
                            
                            # Статистика
                            st.metric("Ср. хеджинг на предложение", 
                                     f"{h['statistics']['mean_hedging_per_sentence']:.2f}")
                            
                            # Индикатор
                            if h['hedging_per_1000'] < 3:
                                st.error("⚠️ Критически низкий уровень хеджинга - сильный сигнал ИИ")
                            elif h['hedging_per_1000'] < 5:
                                st.warning("⚠️ Низкий уровень хеджинга")
                            elif h['hedging_per_1000'] < 7:
                                st.info("Умеренный уровень хеджинга")
                            else:
                                st.success("✅ Хороший уровень хеджинга (человеческий стиль)")
                
                # Вкладка "Новые модули"
                with detail_tabs[3]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'parenthesis' in results:
                            st.subheader("📝 Длинные пояснения в скобках")
                            par = results['parenthesis']
                            
                            st.metric("Всего скобок", par['total_parentheses'])
                            st.metric("Длинных (≥5 слов)", par['long_parentheses'])
                            st.metric("Процент длинных", f"{par['long_percentage']:.1f}%")
                            st.metric("Ср. слов в скобках", f"{par['statistics']['mean_words_in_parentheses']:.1f}")
                            st.metric("Макс. слов в скобках", f"{par['statistics']['max_words_in_parentheses']:.0f}")
                            
                            if par['long_examples']:
                                with st.expander("Примеры длинных пояснений"):
                                    for ex in par['long_examples'][:5]:
                                        st.info(f"({ex['text']}) — {ex['word_count']} слов")
                            
                            # Интерпретация
                            if par['long_percentage'] > 10:
                                st.success("✅ Много длинных пояснений - признак человека")
                            elif par['long_percentage'] > 5:
                                st.info("Умеренное количество длинных пояснений")
                            else:
                                st.warning("⚠️ Мало длинных пояснений - типично для ИИ")
                        
                        if 'apostrophe' in results:
                            st.subheader("🔤 Апострофы 's")
                            ap = results['apostrophe']
                            
                            st.metric("Всего апострофов", ap['apostrophe_count'])
                            st.metric("На 1000 слов", f"{ap['apostrophe_per_1000']:.2f}")
                            st.metric("Ср. на абзац", f"{ap['statistics']['mean_apostrophes_per_paragraph']:.2f}")
                            
                            if ap['possessive_examples']:
                                with st.expander("Примеры притяжательных"):
                                    st.write(", ".join(ap['possessive_examples'][:10]))
                            
                            # Интерпретация
                            if ap['apostrophe_per_1000'] > 1.5:
                                st.error("⚠️ Высокая частота апострофов - признак ИИ")
                            elif ap['apostrophe_per_1000'] > 0.8:
                                st.warning("Умеренная частота апострофов")
                            else:
                                st.success("✅ Низкая частота апострофов - норма для науки")
                    
                    with col2:
                        if 'punctuation' in results:
                            st.subheader("‼️ Пунктуация ! ? ;")
                            pu = results['punctuation']
                            
                            st.metric("Восклицательные знаки", f"{pu['exclamation_count']} ({pu['exclamation_per_1000']:.2f}/1000)")
                            st.metric("Вопросительные знаки", f"{pu['question_count']} ({pu['question_per_1000']:.2f}/1000)")
                            st.metric("Точки с запятой", f"{pu['semicolon_count']} ({pu['semicolon_per_1000']:.2f}/1000)")
                            
                            if pu['semicolon_contexts']:
                                with st.expander("Контексты использования ;"):
                                    for ctx in pu['semicolon_contexts'][:5]:
                                        if ctx['is_contrast']:
                                            st.info(f"Короткое после длинного: ...{ctx['context']}...")
                                        else:
                                            st.write(f"...{ctx['context']}...")
                            
                            # Интерпретация
                            if pu['semicolon_per_1000'] > 0.3:
                                st.success("✅ Активное использование ; - признак человека")
                            elif pu['exclamation_per_1000'] > 0.1 or pu['question_per_1000'] > 0.2:
                                st.success("✅ Разнообразие пунктуации - признак человека")
                            else:
                                st.warning("⚠️ Однообразная пунктуация - типично для ИИ")
                        
                        if 'enumeration' in results:
                            st.subheader("🔢 Перечисления X, Y, and Z")
                            en = results['enumeration']
                            
                            st.metric("Всего перечислений", en['three_item_count'])
                            st.metric("На 1000 предложений", f"{en['three_item_per_1000_sentences']:.1f}")
                            st.metric("С specifically", en['specifically_count'])
                            st.metric("Макс. в предложении", f"{en['statistics']['max_enumerations_in_sentence']:.0f}")
                            
                            if en['examples']:
                                with st.expander("Примеры перечислений"):
                                    for ex in en['examples'][:10]:
                                        st.write(f"- {ex}")
                            
                            # Интерпретация
                            if en['three_item_per_1000_sentences'] > 8:
                                st.error("⚠️ Очень частые строгие перечисления - признак ИИ")
                            elif en['three_item_per_1000_sentences'] > 4:
                                st.warning("Умеренная частота перечислений")
                            else:
                                st.success("✅ Нормальная частота перечислений")
                
                # Вкладка "Семантика"
                with detail_tabs[4]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'paragraph' in results:
                            st.subheader("📑 Абзацный анализ")
                            p = results['paragraph']
                            
                            if 'paragraph_lengths' in p and p['paragraph_lengths']:
                                st.metric("Ср. длина абзаца", f"{p['statistics']['mean_paragraph_length']:.0f} слов")
                                st.metric("Медианная длина", f"{p['statistics']['median_paragraph_length']:.0f}")
                                st.metric("Std длина", f"{p['statistics']['std_paragraph_length']:.1f}")
                            
                            if 'intra_paragraph_similarity' in p:
                                st.metric("Внутриабзацная похожесть", f"{p['intra_paragraph_similarity']:.3f}")
                                st.metric("Межабзацная похожесть", f"{p.get('inter_paragraph_similarity', 0):.3f}")
                                st.metric("Плавность переходов", f"{p.get('transition_smoothness', 0):.3f}")
                                
                                if p.get('intra_paragraph_similarity', 0) > 0.72:
                                    st.error("⚠️ Очень высокая внутриабзацная похожесть - ИИ")
                            
                            st.info(p.get('note', ''))
                            
                            # Гистограмма длин абзацев
                            if 'paragraph_lengths' in p and p['paragraph_lengths']:
                                fig = go.Figure(data=[
                                    go.Histogram(x=p['paragraph_lengths'], nbinsx=20)
                                ])
                                fig.update_layout(
                                    title="Распределение длины абзацев",
                                    xaxis_title="Длина (слова)",
                                    yaxis_title="Частота",
                                    height=250
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if deep_analysis:
                            if 'perplexity' in results:
                                st.subheader("📊 Perplexity")
                                pp = results['perplexity']
                                
                                if 'mean_perplexity' in pp:
                                    st.metric("Средняя перплексия", f"{pp['mean_perplexity']:.2f}")
                                    st.metric("Медианная", f"{pp['median_perplexity']:.2f}")
                                    st.metric("Мин/Макс", f"{pp['min_perplexity']:.1f} / {pp['max_perplexity']:.1f}")
                                    st.metric("Вариация", f"{pp['perplexity_variance']:.2f}")
                                    
                                    if pp['mean_perplexity'] < 20:
                                        st.error("⚠️ Низкая перплексия - признак ИИ")
                                    elif pp['mean_perplexity'] < 35:
                                        st.warning("Средняя перплексия")
                                    else:
                                        st.success("✅ Высокая перплексия - человеческий текст")
                                    
                                    # График перплексии по сегментам
                                    if 'perplexity_segments' in pp and pp['perplexity_segments']:
                                        segments_df = pd.DataFrame(pp['perplexity_segments'])
                                        fig = go.Figure(data=[
                                            go.Scatter(x=segments_df['segment'], y=segments_df['perplexity'],
                                                      mode='lines+markers')
                                        ])
                                        fig.update_layout(
                                            title="Перплексия по сегментам",
                                            xaxis_title="Сегмент",
                                            yaxis_title="Perplexity",
                                            height=250
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                st.info(pp.get('note', ''))
                            
                            if 'semantic' in results:
                                st.subheader("🔄 Семантическая близость")
                                sm = results['semantic']
                                
                                if 'mean_similarity' in sm:
                                    st.metric("Средняя близость", f"{sm['mean_similarity']:.3f}")
                                    st.metric("Вариация", f"{sm['similarity_variance']:.4f}")
                                    st.metric("Мин/Макс", f"{sm['min_similarity']:.2f} / {sm['max_similarity']:.2f}")
                                    st.metric("Семантических кластеров", sm.get('semantic_clusters', 0))
                                    
                                    if sm['mean_similarity'] > 0.6:
                                        st.error("⚠️ Высокая семантическая близость - ИИ")
                                    elif sm['mean_similarity'] > 0.5:
                                        st.warning("Средняя близость")
                                    else:
                                        st.success("✅ Низкая близость - разнообразие идей")
                                    
                                    # Гистограмма сходств
                                    if 'similarities' in sm and sm['similarities']:
                                        fig = go.Figure(data=[
                                            go.Histogram(x=sm['similarities'][:500], nbinsx=30)
                                        ])
                                        fig.update_layout(
                                            title="Распределение семантических сходств",
                                            xaxis_title="Косинусное сходство",
                                            yaxis_title="Частота",
                                            height=250
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                st.info(sm.get('note', ''))
                        else:
                            st.info("Для семантического анализа включите 'Глубокий анализ' в настройках")
            
            # =================================================================
            # Вкладка 3: Текст с разметкой
            # =================================================================
            with tab_text:
                st.markdown("## 📝 Текст статьи с разметкой")
                
                # Показываем статистику текста
                st.markdown(f"""
                <div class="info-box">
                    <b>Всего символов:</b> {len(text)} | 
                    <b>Предложений:</b> {len(sentences)} | 
                    <b>Абзацев:</b> {len(re.split(r'\n\s*\n', text))}
                </div>
                """, unsafe_allow_html=True)
                
                # Подозрительные места
                with st.expander("⚠️ Подозрительные фрагменты", expanded=True):
                    tabs_highlight = st.tabs(["ИИ-фразы", "Перечисления", "Апострофы", "Скобки"])
                    
                    with tabs_highlight[0]:
                        if 'phrases' in results and results['phrases']['repeated_phrases']:
                            for phrase_data in results['phrases']['repeated_phrases'][:5]:
                                phrase = phrase_data['phrase']
                                st.markdown(f"**{phrase}** (встречается {phrase_data['count']} раз)")
                                # Ищем контекст
                                pattern = re.compile(r'([^.]*?' + re.escape(phrase) + r'[^.]*\.)', re.IGNORECASE)
                                matches = pattern.findall(text[:10000])
                                for match in matches[:2]:
                                    st.markdown(f"> ...{match}...")
                                st.markdown("---")
                    
                    with tabs_highlight[1]:
                        if 'enumeration' in results and results['enumeration']['examples']:
                            for ex in results['enumeration']['examples'][:10]:
                                st.markdown(f"- `{ex}`")
                    
                    with tabs_highlight[2]:
                        if 'apostrophe' in results:
                            if results['apostrophe']['possessive_examples']:
                                st.markdown("**Притяжательные формы:**")
                                st.write(", ".join(results['apostrophe']['possessive_examples'][:20]))
                            if results['apostrophe']['contraction_examples']:
                                st.markdown("**Сокращения:**")
                                st.write(", ".join(results['apostrophe']['contraction_examples'][:10]))
                    
                    with tabs_highlight[3]:
                        if 'parenthesis' in results and results['parenthesis']['long_examples']:
                            for ex in results['parenthesis']['long_examples'][:10]:
                                st.markdown(f"- ({ex['text']}) — *{ex['word_count']} слов*")
                
                # Полный текст
                with st.expander("📄 Полный текст", expanded=False):
                    # Разбиваем на абзацы для читаемости
                    paragraphs = re.split(r'\n\s*\n', text)
                    for i, para in enumerate(paragraphs[:20]):
                        if para.strip():
                            st.markdown(f"**Абзац {i+1}:**")
                            st.markdown(para)
                            st.markdown("---")
                    
                    if len(paragraphs) > 20:
                        st.info(f"Показано 20 из {len(paragraphs)} абзацев")
        
        except Exception as e:
            st.error(f"Ошибка при обработке: {str(e)}")
            st.exception(e)
    
    else:
        # Информация о приложении
        st.info("👆 Загрузите файл и нажмите 'Начать анализ' в левой панели")
        
        with st.expander("ℹ️ О метриках анализа (обновлено 2025-2026)"):
            st.markdown("""
            ### Новые модули анализа:
            
            **1. Длинные пояснения в скобках ()**
            - ИИ редко делает пояснения >5 слов в скобках
            - Высокий процент длинных скобок → признак человека
            
            **2. Пунктуация ! ? ;**
            - Восклицательные знаки редки в науке, но признак человека
            - Точка с запятой (;) для контраста длинного и короткого → человек
            - ИИ избегает редких знаков пунктуации
            
            **3. Апострофы 's**
            - Частое использование (Parkinson's, pathway's) → признак ИИ
            - В научных статьях апострофы редки
            
            **4. Перечисления X, Y, and Z**
            - Строгие трехэлементные перечисления → типичны для ИИ
            - Особенно с "specifically" → сильный сигнал
            
            ### Обновленные веса:
            - Хеджинг остается ключевым маркером (15%)
            - Новые модули суммарно дают 21% (скобки 6%, пунктуация 5%, апострофы 5%, перечисления 5%)
            - Уменьшены веса устаревших маркеров (unicode, тире)
            
            ### Статистика:
            - Для всех метрик теперь доступны среднее, медиана, максимум
            - Распределения для визуального анализа
            - Детальные контексты для каждого маркера
            """)


if __name__ == "__main__":
    main()
