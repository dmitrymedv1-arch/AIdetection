"""
Streamlit приложение для анализа научных статей на предмет использования ИИ
Исправленная версия с учетом совместимости версий и оптимизацией для Streamlit Cloud
Версия 4.0 - Добавлены: улучшенные перечисления, все примеры артефактов, log-prob анализ,
Yule's I, репетитивность-скор, лексическое разнообразие (MTLD/MATTR/HD-D), ML-классификатор
Название: CT(A)I-detector
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
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
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

# Для лексического разнообразия
try:
    import lexical_diversity as lexdiv
    LEXICAL_DIVERSITY_AVAILABLE = True
except ImportError:
    LEXICAL_DIVERSITY_AVAILABLE = False

# Конфигурация страницы
st.set_page_config(
    page_title="CT(A)I-detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Компактный современный дизайн
st.markdown("""
<style>
    /* Компактный современный дизайн */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main > div {
        padding: 1rem;
        background: white;
        border-radius: 15px;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Компактные карточки */
    .compact-metric {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
        font-size: 0.9rem;
    }
    
    .compact-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Горизонтальный скролл для модулей */
    .modules-scroll {
        display: flex;
        overflow-x: auto;
        gap: 0.5rem;
        padding: 0.5rem 0;
        scrollbar-width: thin;
        white-space: nowrap;
    }
    
    .module-badge {
        background: #e9ecef;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        flex-shrink: 0;
    }
    
    .module-badge.active {
        background: #667eea;
        color: white;
    }
    
    /* Заголовок */
    .ctai-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding-bottom: 0;
    }
    
    /* Сетка для метрик */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Индикатор риска компактный */
    .risk-indicator {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .risk-high { background: #fee; color: #c00; }
    .risk-medium { background: #fff3e0; color: #f90; }
    .risk-low { background: #e8f5e9; color: #2e7d32; }
    
    /* Компактные табы */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.25rem;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
    
    /* Убираем лишние отступы */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Компактные expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        padding: 0.5rem;
    }
    
    /* Скрываем лишнюю информацию */
    .element-container:has(.info-box) {
        display: none;
    }
    
    /* Компактные графики */
    .js-plotly-plot {
        margin: 0 !important;
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
            'all_suspicious_chunks': [],  # Все найденные примеры
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
                context = text[max(0, i-30):min(len(text), i+30)]
                chunk = {
                    'char': char,
                    'context': context,
                    'type': 'superscript/subscript'
                }
                results['suspicious_chunks'].append(chunk)
                results['all_suspicious_chunks'].append(chunk)
            
            # Проверка fullwidth цифр
            elif char in self.fullwidth_digits:
                results['fullwidth_count'] += 1
                context = text[max(0, i-30):min(len(text), i+30)]
                chunk = {
                    'char': char,
                    'context': context,
                    'type': 'fullwidth digit'
                }
                results['suspicious_chunks'].append(chunk)
                results['all_suspicious_chunks'].append(chunk)
            
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
            'all_dash_sentences': [],  # Все предложения с тире
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
            
            sent_data = {
                'sentence': sent[:200] + '...' if len(sent) > 200 else sent,
                'dash_count': dash_count,
                'word_count': word_count
            }
            
            # Сохраняем все предложения с тире
            if dash_count > 0:
                results['all_dash_sentences'].append(sent_data)
            
            if dash_count >= 2:
                results['sentences_with_multiple_dashes'].append(sent_data)
            
            if is_heavy:
                results['heavy_sentences'].append(sent_data)
                if len(results['examples']) < 5:
                    results['examples'].append(sent[:150])
        
        if sentences:
            results['percentage_heavy'] = (len(results['heavy_sentences']) / len(sentences)) * 100
            
            # Статистика распределения
            if dash_counts:
                results['statistics']['mean_dashes_per_sentence'] = float(np.mean(dash_counts))
                results['statistics']['median_dashes_per_sentence'] = float(np.median(dash_counts))
                results['statistics']['max_dashes_in_sentence'] = float(np.max(dash_counts))
                results['statistics']['distribution'] = dash_counts[:100]
        
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
            'pathway', 'pathways',
            'signaling', 'signals',
            'Collectively',
            'manifest',
            'paradigm',
            
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
            'all_phrase_occurrences': [],  # Все вхождения фраз с контекстом
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
        
        # Подсчет каждой фразы и сбор всех вхождений
        for phrase in self.ai_phrases:
            count = text_lower.count(phrase.lower())
            if count > 0:
                results['phrase_counts'][phrase] = count
                
                # Находим все вхождения с контекстом
                pattern = re.compile(r'([^.]*?' + re.escape(phrase) + r'[^.]*\.)', re.IGNORECASE)
                matches = pattern.findall(text[:50000])  # Ограничиваем для производительности
                for match in matches[:20]:  # Не больше 20 на фразу
                    results['all_phrase_occurrences'].append({
                        'phrase': phrase,
                        'context': match.strip()
                    })
        
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
        for sent in sentences[:100]:
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
    """Анализ вариативности длины предложений (уровень 2) - обновлен с Yule's I"""
    
    def analyze(self, sentences: List[str]) -> Dict:
        """Анализирует burstiness текста с использованием Yule's I"""
        results = {
            'sentence_lengths': [],
            'mean_length': 0,
            'std_length': 0,
            'cv': 0,
            'iqr': 0,
            'yules_i': 0,  # Yule's I характеристика
            'sic': 0,      # Syntactic Irregularity Coefficient
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
        
        # Yule's I (мера дисперсии, устойчивая к длине текста)
        # I = (σ² / μ) * 10000
        variance = np.var(sent_lengths)
        if results['mean_length'] > 0:
            results['yules_i'] = float((variance / results['mean_length']) * 10000)
        
        # Syntactic Irregularity Coefficient (SIC) - простая метрика
        # Отношение максимальной длины к минимальной с поправкой
        if results['statistics']['min_length'] > 0:
            results['sic'] = float(results['statistics']['max_length'] / results['statistics']['min_length'])
        
        # Дополнительная статистика
        results['statistics']['median_length'] = float(np.median(sent_lengths))
        results['statistics']['max_length'] = float(np.max(sent_lengths))
        results['statistics']['min_length'] = float(np.min(sent_lengths))
        results['statistics']['percentile_25'] = float(q25)
        results['statistics']['percentile_75'] = float(q75)
        results['statistics']['distribution'] = sent_lengths[:200]
        
        # Вероятностная оценка burstiness на основе Yule's I
        risk_score = 0
        confidence = 0.5
        
        # Нормальные значения Yule's I для научных текстов: ~200-800
        if results['yules_i'] < 150:
            results['burstiness'] = 'very_low'
            risk_score = 3
            confidence = 0.9
        elif results['yules_i'] < 250:
            results['burstiness'] = 'low'
            risk_score = 2
            confidence = 0.7
        elif results['yules_i'] < 500:
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


class RepetitivenessAnalyzer:
    """Анализ повторяемости N-грамм (новый модуль)"""
    
    def __init__(self):
        self.n_values = [3, 4]  # 3-граммы и 4-граммы
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует повторяемость фраз в тексте"""
        results = {
            'ngram_repetition_scores': {},
            'unique_ngram_ratios': {},
            'repeated_phrases': [],
            'all_repetitions': [],  # Все найденные повторы
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'statistics': {
                'mean_repetition_rate': 0,
                'max_repetition_rate': 0,
                'repetition_distribution': {}
            }
        }
        
        if not text or len(text.split()) < 50:
            return results
        
        words = text.lower().split()
        
        for n in self.n_values:
            if len(words) < n:
                continue
                
            # Собираем все N-граммы
            ngrams = []
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
            
            # Считаем частоты
            ngram_counts = Counter(ngrams)
            total_ngrams = len(ngrams)
            unique_ngrams = len(ngram_counts)
            
            # Доля уникальных (чем меньше, тем больше повторов)
            unique_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 1.0
            
            # Собираем повторяющиеся фразы (встречаются >= 3 раз)
            repeated = [(ngram, count) for ngram, count in ngram_counts.items() if count >= 3]
            repeated.sort(key=lambda x: x[1], reverse=True)
            
            # Сохраняем результаты
            results['ngram_repetition_scores'][f'{n}gram'] = 1.0 - unique_ratio  # Чем ближе к 1, тем больше повторов
            results['unique_ngram_ratios'][f'{n}gram'] = unique_ratio
            
            # Сохраняем топ повторений
            for ngram, count in repeated[:20]:
                results['all_repetitions'].append({
                    'ngram': ngram,
                    'count': count,
                    'type': f'{n}-gram'
                })
            
            results['statistics']['repetition_distribution'][f'{n}gram'] = [c for _, c in repeated[:50]]
        
        if results['ngram_repetition_scores']:
            results['statistics']['mean_repetition_rate'] = float(np.mean(list(results['ngram_repetition_scores'].values())))
            results['statistics']['max_repetition_rate'] = float(np.max(list(results['ngram_repetition_scores'].values())))
        
        # Оценка риска
        risk_score = 0
        confidence = 0.5
        
        # Проверяем 3-граммы (более чувствительны к повторам)
        if '3gram' in results['ngram_repetition_scores']:
            rep_rate = results['ngram_repetition_scores']['3gram']
            if rep_rate > 0.25:
                risk_score += 3
                confidence = min(confidence + 0.3, 1.0)
            elif rep_rate > 0.15:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif rep_rate > 0.08:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
        
        # Проверяем 4-граммы
        if '4gram' in results['ngram_repetition_scores']:
            rep_rate = results['ngram_repetition_scores']['4gram']
            if rep_rate > 0.15:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif rep_rate > 0.08:
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


class LexicalDiversityAnalyzer:
    """Анализ лексического разнообразия (MTLD, MATTR, HD-D)"""
    
    def __init__(self):
        self.available = LEXICAL_DIVERSITY_AVAILABLE
    
    def analyze(self, text: str) -> Dict:
        """Анализирует лексическое разнообразие текста"""
        results = {
            'ttr': 0,  # Type-Token Ratio (базовый)
            'mtld': 0,  # Measure of Textual Lexical Diversity
            'mattr': 0,  # Moving Average Type-Token Ratio
            'hdd': 0,    # Hypergeometric Distribution Diversity
            'hapax_legomena': 0,  # Слова, встретившиеся 1 раз
            'hapax_dislegomena': 0,  # Слова, встретившиеся 2 раза
            'hapax_ratio': 0,
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'note': 'Lexical diversity analysis requires lexical-diversity library'
        }
        
        if not text or len(text.split()) < 50:
            return results
        
        words = text.lower().split()
        unique_words = set(words)
        
        # Базовый TTR
        results['ttr'] = len(unique_words) / len(words) if words else 0
        
        # Считаем hapax legomena (слова, встретившиеся 1 раз)
        word_counts = Counter(words)
        results['hapax_legomena'] = sum(1 for count in word_counts.values() if count == 1)
        results['hapax_dislegomena'] = sum(1 for count in word_counts.values() if count == 2)
        results['hapax_ratio'] = results['hapax_legomena'] / len(words) if words else 0
        
        # Если доступна библиотека lexical-diversity, используем продвинутые метрики
        if self.available:
            try:
                # Преобразуем в формат для lexical-diversity
                text_for_lex = ' '.join(words)
                
                # MTLD
                results['mtld'] = lexdiv.mtld(text_for_lex)
                
                # MATTR (window size = 50)
                results['mattr'] = lexdiv.mattr(text_for_lex, window_size=50)
                
                # HD-D
                results['hdd'] = lexdiv.hdd(text_for_lex, sample_size=42)
                
                results['note'] = 'Full lexical diversity analysis'
            except Exception as e:
                results['note'] = f'Error in advanced lexical analysis: {str(e)}'
        
        # Оценка риска на основе лексического разнообразия
        risk_score = 0
        confidence = 0.5
        
        # Низкое лексическое разнообразие - признак ИИ
        if results['mtld'] > 0:
            if results['mtld'] < 40:
                risk_score += 3
                confidence = min(confidence + 0.3, 1.0)
            elif results['mtld'] < 60:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif results['mtld'] < 80:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
        else:
            # Fallback на TTR
            if results['ttr'] < 0.4:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif results['ttr'] < 0.5:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
        
        # Низкая доля hapax legomena - признак шаблонности
        if results['hapax_ratio'] < 0.5:
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


class LogProbAnalyzer:
    """Анализ log-probabilities через открытые модели"""
    
    def __init__(self):
        self.available = TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
        
        if self.available:
            try:
                # Используем небольшую модель Qwen2.5-7B (урезанную для памяти)
                model_name = "Qwen/Qwen2.5-7B-Instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                try:
                    # Fallback на более легкую модель
                    model_name = "microsoft/Phi-3.5-mini-instruct"
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except Exception as e2:
                    self.available = False
    
    def calculate_log_probs(self, text: str) -> Dict:
        """Вычисляет средние log-probabilities для текста"""
        if not self.available or len(text) < 100:
            return {}
        
        try:
            # Токенизируем
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Вычисляем logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Вычисляем log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Берем вероятности для реальных токенов
                token_log_probs = []
                for i in range(inputs['input_ids'].shape[1] - 1):
                    token_id = inputs['input_ids'][0, i+1]
                    token_log_prob = log_probs[0, i, token_id].item()
                    token_log_probs.append(token_log_prob)
                
                # Статистики
                results = {
                    'mean_log_prob': float(np.mean(token_log_probs)),
                    'median_log_prob': float(np.median(token_log_probs)),
                    'std_log_prob': float(np.std(token_log_probs)),
                    'min_log_prob': float(np.min(token_log_probs)),
                    'max_log_prob': float(np.max(token_log_probs))
                }
                
                return results
        except Exception as e:
            return {}
    
    def analyze(self, text: str) -> Dict:
        """Анализирует текст через log-probabilities"""
        results = {
            'log_prob_stats': {},
            'mean_log_prob': 0,
            'perplexity_estimate': 0,
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'note': 'Log-probability analysis requires transformers'
        }
        
        if not self.available or len(text) < 100:
            results['note'] = 'Log-probability analysis unavailable or text too short'
            return results
        
        log_prob_data = self.calculate_log_probs(text)
        if log_prob_data:
            results['log_prob_stats'] = log_prob_data
            results['mean_log_prob'] = log_prob_data['mean_log_prob']
            
            # Приблизительная перплексия из log-prob
            results['perplexity_estimate'] = float(np.exp(-log_prob_data['mean_log_prob']))
            
            # Оценка риска
            risk_score = 0
            confidence = 0.5
            
            # Более высокие log-probabilities (менее отрицательные) - признак ИИ
            if results['mean_log_prob'] > -2.0:
                risk_score = 3
                confidence = 0.9
            elif results['mean_log_prob'] > -3.0:
                risk_score = 2
                confidence = 0.7
            elif results['mean_log_prob'] > -4.0:
                risk_score = 1
                confidence = 0.6
            
            results['risk_score'] = risk_score
            results['confidence'] = confidence
            
            if risk_score >= 3:
                results['risk_level'] = 'high'
            elif risk_score >= 2:
                results['risk_level'] = 'medium'
            elif risk_score >= 1:
                results['risk_level'] = 'low'
        
        return results


class MLClassifier:
    """ML-классификатор на основе BERT/RoBERTa (fine-tuned)"""
    
    def __init__(self):
        self.available = TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
        
        if self.available:
            try:
                # Используем предобученный детектор ИИ (если доступен)
                # В реальном проекте здесь будет ваш fine-tuned модель
                model_name = "roberta-base-openai-detector"  # Пример
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            except:
                # Если нет готового, используем заглушку
                self.available = False
    
    def analyze(self, text: str) -> Dict:
        """Анализирует текст с помощью ML-модели"""
        results = {
            'ml_score': 0,
            'ml_probability': 0,
            'ml_confidence': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'note': 'ML classifier not available'
        }
        
        if not self.available or len(text) < 100:
            return results
        
        try:
            # Токенизируем
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Предполагаем бинарную классификацию: 0 - human, 1 - AI
                ai_prob = probs[0, 1].item()
                
                results['ml_score'] = ai_prob * 100
                results['ml_probability'] = ai_prob
                results['ml_confidence'] = max(probs[0]).item()
                
                # Оценка риска
                risk_score = 0
                confidence = results['ml_confidence']
                
                if ai_prob > 0.8:
                    risk_score = 3
                elif ai_prob > 0.6:
                    risk_score = 2
                elif ai_prob > 0.4:
                    risk_score = 1
                
                results['risk_score'] = risk_score
                results['confidence'] = confidence
                
                if risk_score >= 3:
                    results['risk_level'] = 'high'
                elif risk_score >= 2:
                    results['risk_level'] = 'medium'
                elif risk_score >= 1:
                    results['risk_level'] = 'low'
                
                results['note'] = 'ML classifier analysis'
                
        except Exception as e:
            results['note'] = f'Error in ML classification: {str(e)}'
        
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
            'all_parentheses': [],  # Все найденные скобки
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
        
        for match in matches:
            words = match.split()
            word_count = len(words)
            word_counts.append(word_count)
            
            parenthesis_data = {
                'text': match[:200] + '...' if len(match) > 200 else match,
                'word_count': word_count
            }
            results['all_parentheses'].append(parenthesis_data)
            
            if word_count >= self.min_words_for_long:
                results['long_parentheses'] += 1
                if len(results['long_examples']) < 20:
                    results['long_examples'].append(parenthesis_data)
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
            risk_score = 0
            confidence = 0.8
        elif results['long_percentage'] > 8:
            risk_score = 1
            confidence = 0.6
        elif results['long_percentage'] > 3:
            risk_score = 2
            confidence = 0.5
        else:
            risk_score = 3
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
            'all_semicolon_contexts': [],  # Все контексты с ;
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
        for pos in semicolon_positions[:50]:  # Ограничиваем для производительности
            context = text[max(0, pos-60):min(len(text), pos+60)]
            # Проверяем, является ли это разделением длинного и короткого предложения
            before = context[:60].split(';')[0] if ';' in context[:60] else ''
            after = context[60:].split(';')[0] if ';' in context[60:] else ''
            
            before_words = len(before.split())
            after_words = len(after.split())
            
            context_data = {
                'context': context.replace('\n', ' ').strip(),
                'before_words': before_words,
                'after_words': after_words,
                'is_contrast': after_words < before_words * 0.3
            }
            results['all_semicolon_contexts'].append(context_data)
            if len(results['semicolon_contexts']) < 10:
                results['semicolon_contexts'].append(context_data)
        
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
            risk_score -= 1
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
            'all_apostrophes': [],  # Все найденные апострофы
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
        for match in apostrophe_matches:
            match_data = match
            results['all_apostrophes'].append(match_data)
            
            if match.endswith("'s") and len(match) > 2:
                base_word = match[:-2]
                if base_word.isalpha() and len(results['possessive_examples']) < 50:
                    results['possessive_examples'].append(match)
            elif any(cont in match for cont in ["'t", "'re", "'ve", "'ll", "'m", "'d"]):
                if len(results['contraction_examples']) < 30:
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
    """Анализ строгих перечислений (улучшенная версия)"""
    
    def __init__(self):
        # Улучшенный паттерн для перечислений: словосочетание, словосочетание, and словосочетание
        self.enumeration_pattern = r'(\b[^,]+(?:,\s+[^,]+)+,\s+and\s+[^,]+\.?)'
        # Паттерн для перечислений с specifically
        self.specifically_pattern = r'\bspecifically\s+[^,]+,\s+[^,]+,\s+and\s+[^,]+'
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует строгие перечисления из трех и более элементов"""
        results = {
            'three_item_count': 0,
            'three_item_per_1000_sentences': 0,
            'specifically_count': 0,
            'examples': [],
            'all_enumerations': [],  # Все найденные перечисления
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
        
        # Поиск трехэлементных перечислений (улучшенный паттерн)
        enumeration_matches = re.findall(self.enumeration_pattern, text, re.IGNORECASE | re.DOTALL)
        
        # Фильтруем слишком длинные совпадения (вероятно, целые предложения)
        valid_enumerations = []
        for match in enumeration_matches:
            # Очищаем от лишних пробелов и переносов
            clean_match = re.sub(r'\s+', ' ', match).strip()
            word_count = len(clean_match.split())
            # Перечисление должно быть разумной длины (не целое предложение)
            if 5 <= word_count <= 30:
                valid_enumerations.append(clean_match)
        
        results['three_item_count'] = len(valid_enumerations)
        results['all_enumerations'] = valid_enumerations
        
        # Поиск перечислений с specifically
        specifically_matches = re.findall(self.specifically_pattern, text, re.IGNORECASE)
        results['specifically_count'] = len(specifically_matches)
        
        # Нормировка на 1000 предложений
        if total_sentences > 0:
            results['three_item_per_1000_sentences'] = (results['three_item_count'] * 1000) / total_sentences
        
        # Собираем примеры
        results['examples'] = valid_enumerations[:20]
        
        # Статистика по предложениям
        enumerations_per_sentence = []
        for sent in sentences[:100]:
            sent_enumerations = len(re.findall(self.enumeration_pattern, sent, re.IGNORECASE))
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
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
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
        
        results['paragraphs'] = paragraphs[:15]
        
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
            para_embeddings = self.model.encode(paragraphs[:25])
            
            # Внутриабзацная похожесть
            intra_similarities = []
            for para in paragraphs[:8]:
                para_sents = [s for s in re.split(r'[.!?]+', para) if len(s.split()) > 5]
                if len(para_sents) >= 3:
                    sent_embeddings = self.model.encode(para_sents[:10])
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
            
            # Плавность переходов
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
            
            if results['intra_paragraph_similarity'] > 0.72:
                risk_score += 3
                confidence = min(confidence + 0.3, 1.0)
            elif results['intra_paragraph_similarity'] > 0.65:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif results['intra_paragraph_similarity'] > 0.6:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
            
            if results['transition_smoothness'] > 0.65:
                risk_score += 2
                confidence = min(confidence + 0.2, 1.0)
            elif results['transition_smoothness'] > 0.55:
                risk_score += 1
                confidence = min(confidence + 0.1, 1.0)
            
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
    
    def calculate_perplexity(self, text: str) -> float:
        """Вычисляет перплексию текста"""
        if not self.available or not text:
            return 0
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
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
            'all_perplexities': [],  # Все значения перплексии
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
            words = text.split()
            segment_size = 500
            segments = [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]
            segments = segments[:20]
            
            perplexities = []
            
            for i, segment in enumerate(segments):
                if len(segment.split()) < 50:
                    continue
                
                ppl = self.calculate_perplexity(segment)
                if ppl > 0:
                    perplexities.append(ppl)
                    results['perplexities'].append(ppl)
                    results['all_perplexities'].append(ppl)
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
                risk_score = 3
                confidence = 0.8
            elif results['mean_perplexity'] < 25:
                risk_score = 2
                confidence = 0.7
            elif results['mean_perplexity'] < 40:
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
            'all_similarities': [],  # Все значения сходств
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
            sample_sentences = [s for s in sentences if len(s.split()) > 5][:50]
            
            if len(sample_sentences) < 5:
                return results
            
            embeddings = self.model.encode(sample_sentences)
            
            similarities = []
            n = len(embeddings)
            
            for i in range(n):
                for j in range(i+1, n):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    similarities.append(sim)
            
            if similarities:
                results['similarities'] = similarities[:1000]
                results['all_similarities'] = similarities
                results['mean_similarity'] = float(np.mean(similarities))
                results['similarity_variance'] = float(np.var(similarities))
                results['min_similarity'] = float(np.min(similarities))
                results['max_similarity'] = float(np.max(similarities))
                results['statistics']['distribution'] = similarities[:500]
                
                # Простая кластеризация
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
                risk_score = 3
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
            'unicode': 0.03,      # 3%
            'dashes': 0.03,        # 3%
            'phrases': 0.08,       # 8%
            'burstiness': 0.06,    # 6%
            'grammar': 0.08,       # 8%
            'hedging': 0.12,       # 12% (ключевой)
            'paragraph': 0.08,     # 8%
            'perplexity': 0.06,    # 6%
            'semantic': 0.06,      # 6%
            'parenthesis': 0.05,   # 5%
            'punctuation': 0.04,   # 4%
            'apostrophe': 0.04,    # 4%
            'enumeration': 0.04,   # 4%
            'repetitiveness': 0.07, # 7% (новый)
            'lexical_diversity': 0.07, # 7% (новый)
            'log_prob': 0.05,      # 5% (новый)
            'ml_classifier': 0.04  # 4% (новый)
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
            if len(content) > 100:
                return content
            
            return None
        except Exception as e:
            return None
    
    @staticmethod
    def split_sentences_simple(text: str) -> List[str]:
        """Простая сегментация на предложения без spacy"""
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
    st.markdown('<h1 class="ctai-header">CT(A)I-detector</h1>', unsafe_allow_html=True)
    
    # Левая боковая панель
    with st.sidebar:
        st.markdown("### 📁 Загрузка")
        uploaded_file = st.file_uploader("", type=['docx', 'doc'])
        
        if uploaded_file:
            st.caption(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        st.markdown("### ⚙️ Настройки")
        cut_references = st.checkbox("Обрезать References", value=True)
        deep_analysis = st.checkbox("Глубокий анализ", value=False)
        
        # Компактный список модулей горизонтальным скроллом
        st.markdown("### 🧩 Модули")
        modules_html = """
        <div class="modules-scroll">
            <span class="module-badge active">📝 Скобки</span>
            <span class="module-badge active">‼️ Пунктуация</span>
            <span class="module-badge active">🔤 Апострофы</span>
            <span class="module-badge active">🔢 Перечисления</span>
            <span class="module-badge active">📊 Хеджинг</span>
            <span class="module-badge active">🔄 Повторы</span>
            <span class="module-badge active">📚 Лексика</span>
            <span class="module-badge active">📈 Burstiness</span>
            <span class="module-badge active">🤖 ML</span>
            <span class="module-badge active">📉 Log-prob</span>
        </div>
        """
        st.markdown(modules_html, unsafe_allow_html=True)
        
        if st.button("🚀 Анализ", use_container_width=True):
            st.session_state['analyze'] = True
    
    # Основная область
    if uploaded_file is not None and st.session_state.get('analyze', False):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Чтение документа
            status_text.text("Чтение...")
            progress_bar.progress(5)
            
            if uploaded_file.name.endswith('.docx'):
                text = DocumentProcessor.read_docx(uploaded_file)
            else:
                text = DocumentProcessor.read_doc(uploaded_file)
            
            if not text or len(text.strip()) < 100:
                st.warning("Файл слишком мал")
                return
            
            # Обрезка и предобработка
            if cut_references:
                text = ReferenceCutoff.cut_at_references(text)
            text = DocumentProcessor.preprocess(text)
            sentences = DocumentProcessor.split_sentences_simple(text)
            
            # Хранилище результатов
            results = {}
            
            # Запуск всех анализаторов
            with st.spinner("Анализ..."):
                # Базовые модули
                results['unicode'] = UnicodeArtifactDetector().analyze(text)
                progress_bar.progress(10)
                
                results['dashes'] = DashAnalyzer().analyze(sentences)
                progress_bar.progress(15)
                
                results['phrases'] = AIPhraseDetector().analyze(text, sentences)
                progress_bar.progress(20)
                
                results['burstiness'] = BurstinessAnalyzer().analyze(sentences)
                progress_bar.progress(25)
                
                results['grammar'] = GrammarAnalyzer().analyze(text)
                progress_bar.progress(30)
                
                results['hedging'] = HedgingAnalyzer().analyze(text)
                progress_bar.progress(35)
                
                # Новые модули
                results['parenthesis'] = ParenthesisAnalyzer().analyze(text)
                progress_bar.progress(40)
                
                results['punctuation'] = PunctuationAnalyzer().analyze(text, sentences)
                progress_bar.progress(45)
                
                results['apostrophe'] = ApostropheAnalyzer().analyze(text)
                progress_bar.progress(50)
                
                results['enumeration'] = EnumerationAnalyzer().analyze(text, sentences)
                progress_bar.progress(55)
                
                results['paragraph'] = ParagraphAnalyzer().analyze(text, sentences)
                progress_bar.progress(60)
                
                # Новые улучшенные модули
                results['repetitiveness'] = RepetitivenessAnalyzer().analyze(text, sentences)
                progress_bar.progress(65)
                
                results['lexical_diversity'] = LexicalDiversityAnalyzer().analyze(text)
                progress_bar.progress(70)
                
                if deep_analysis:
                    results['log_prob'] = LogProbAnalyzer().analyze(text)
                    progress_bar.progress(80)
                    
                    results['perplexity'] = PerplexityAnalyzer().analyze(text)
                    progress_bar.progress(85)
                    
                    results['semantic'] = SemanticAnalyzer().analyze(sentences)
                    progress_bar.progress(90)
                    
                    results['ml_classifier'] = MLClassifier().analyze(text)
                    progress_bar.progress(95)
            
            # Интегральная оценка
            scorer = IntegratedRiskScorer()
            integrated = scorer.calculate(results)
            
            progress_bar.progress(100)
            status_text.text("✅ Готово!")
            
            # Компактный дашборд
            col1, col2, col3, col4 = st.columns([2,1,1,1])
            
            with col1:
                risk_class = f"risk-{integrated['risk_level'].split('_')[0]}"
                st.markdown(f"""
                <div class="risk-indicator {risk_class}">
                    <span>🎯 {integrated['final_score']:.1f}</span>
                    <span>{integrated['risk_level'].replace('_', ' ').title()}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Предложений", len(sentences))
            with col3:
                st.metric("Слов", len(text.split()))
            with col4:
                st.metric("Модулей", len(integrated['module_scores']))
            
            # Основные метрики в компактной сетке
            st.markdown("### 📊 Ключевые метрики")
            
            metrics_html = '<div class="metrics-grid">'
            
            # Хеджинг
            if 'hedging' in results:
                h = results['hedging']
                color = "#2e7d32" if h['hedging_per_1000'] > 5 else "#c00"
                metrics_html += f"""
                <div class="compact-metric">
                    <b>Хеджинг</b><br>
                    <span style="color:{color}; font-size:1.2rem;">{h['hedging_per_1000']:.1f}</span>/1000
                </div>
                """
            
            # Длинные скобки
            if 'parenthesis' in results:
                p = results['parenthesis']
                color = "#2e7d32" if p['long_percentage'] > 8 else "#c00"
                metrics_html += f"""
                <div class="compact-metric">
                    <b>Длинные скобки</b><br>
                    <span style="color:{color};">{p['long_percentage']:.1f}%</span>
                </div>
                """
            
            # Перечисления
            if 'enumeration' in results:
                e = results['enumeration']
                color = "#c00" if e['three_item_per_1000_sentences'] > 5 else "#2e7d32"
                metrics_html += f"""
                <div class="compact-metric">
                    <b>Перечисления</b><br>
                    <span style="color:{color};">{e['three_item_per_1000_sentences']:.1f}</span>/1000
                </div>
                """
            
            # Повторы
            if 'repetitiveness' in results:
                r = results['repetitiveness']
                rep_rate = r.get('ngram_repetition_scores', {}).get('3gram', 0) * 100
                color = "#c00" if rep_rate > 15 else "#2e7d32"
                metrics_html += f"""
                <div class="compact-metric">
                    <b>Повторы 3-грамм</b><br>
                    <span style="color:{color};">{rep_rate:.1f}%</span>
                </div>
                """
            
            # Лексическое разнообразие
            if 'lexical_diversity' in results:
                l = results['lexical_diversity']
                mtld = l.get('mtld', 0)
                if mtld > 0:
                    color = "#2e7d32" if mtld > 60 else "#c00"
                    metrics_html += f"""
                    <div class="compact-metric">
                        <b>MTLD</b><br>
                        <span style="color:{color};">{mtld:.1f}</span>
                    </div>
                    """
            
            metrics_html += '</div>'
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Вкладки с деталями
            tabs = st.tabs(["Артефакты", "Перечисления", "Повторы", "Лексика", "Все примеры"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'unicode' in results:
                        with st.expander(f"Unicode-артефакты ({len(results['unicode']['all_suspicious_chunks'])} найдено)"):
                            for chunk in results['unicode']['all_suspicious_chunks'][:50]:
                                st.code(f"'{chunk['char']}' → ...{chunk['context'][:100]}...")
                    
                    if 'dashes' in results:
                        with st.expander(f"Тире ({len(results['dashes']['all_dash_sentences'])} найдено)"):
                            for item in results['dashes']['all_dash_sentences'][:30]:
                                st.info(f"Тире: {item['dash_count']} → {item['sentence'][:150]}")
                
                with col2:
                    if 'punctuation' in results:
                        with st.expander(f"Точки с запятой ({len(results['punctuation']['all_semicolon_contexts'])} найдено)"):
                            for ctx in results['punctuation']['all_semicolon_contexts'][:30]:
                                st.code(f"{'📏 Контраст' if ctx['is_contrast'] else '➡️'} ...{ctx['context'][:150]}...")
                    
                    if 'apostrophe' in results:
                        with st.expander(f"Апострофы ({len(results['apostrophe']['all_apostrophes'])} найдено)"):
                            st.write(", ".join(results['apostrophe']['all_apostrophes'][:100]))
            
            with tabs[1]:
                if 'enumeration' in results:
                    st.subheader(f"Найдено перечислений: {len(results['enumeration']['all_enumerations'])}")
                    for i, enum in enumerate(results['enumeration']['all_enumerations'][:100]):
                        st.markdown(f"{i+1}. `{enum}`")
                
                if 'phrases' in results:
                    with st.expander(f"ИИ-фразы ({len(results['phrases']['all_phrase_occurrences'])} вхождений)"):
                        for occ in results['phrases']['all_phrase_occurrences'][:50]:
                            st.markdown(f"**{occ['phrase']}** → {occ['context'][:150]}")
            
            with tabs[2]:
                if 'repetitiveness' in results:
                    st.subheader("Повторяющиеся фразы")
                    for rep in results['repetitiveness']['all_repetitions'][:50]:
                        st.markdown(f"**{rep['ngram']}** — {rep['count']} раз(а)")
            
            with tabs[3]:
                if 'lexical_diversity' in results:
                    l = results['lexical_diversity']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("TTR", f"{l['ttr']:.3f}")
                        st.metric("MTLD", f"{l.get('mtld', 0):.1f}")
                        st.metric("MATTR", f"{l.get('mattr', 0):.3f}")
                    with col2:
                        st.metric("HD-D", f"{l.get('hdd', 0):.3f}")
                        st.metric("Hapax Ratio", f"{l['hapax_ratio']:.3f}")
                        st.metric("Уникальных слов", f"{len(set(l.get('words', [])))}")
            
            with tabs[4]:
                st.subheader("Все найденные примеры по категориям")
                
                # Скобки
                if 'parenthesis' in results and results['parenthesis']['all_parentheses']:
                    with st.expander(f"📝 Скобки ({len(results['parenthesis']['all_parentheses'])} найдено)"):
                        for p in results['parenthesis']['all_parentheses'][:100]:
                            st.markdown(f"({p['text']}) — *{p['word_count']} слов*")
                
                # Перечисления (дублируем для полноты)
                if 'enumeration' in results and results['enumeration']['all_enumerations']:
                    with st.expander(f"🔢 Перечисления ({len(results['enumeration']['all_enumerations'])} найдено)"):
                        for enum in results['enumeration']['all_enumerations'][:50]:
                            st.code(enum)
                
                # Апострофы
                if 'apostrophe' in results and results['apostrophe']['all_apostrophes']:
                    with st.expander(f"🔤 Апострофы ({len(results['apostrophe']['all_apostrophes'])} найдено)"):
                        st.write(", ".join(results['apostrophe']['all_apostrophes'][:200]))
                
                # ИИ-фразы
                if 'phrases' in results and results['phrases']['all_phrase_occurrences']:
                    with st.expander(f"🤖 ИИ-фразы ({len(results['phrases']['all_phrase_occurrences'])} вхождений)"):
                        for occ in results['phrases']['all_phrase_occurrences'][:100]:
                            st.markdown(f"**{occ['phrase']}** → {occ['context'][:150]}")
                
                # Точки с запятой
                if 'punctuation' in results and results['punctuation']['all_semicolon_contexts']:
                    with st.expander(f"‼️ Точки с запятой ({len(results['punctuation']['all_semicolon_contexts'])} найдено)"):
                        for ctx in results['punctuation']['all_semicolon_contexts'][:50]:
                            st.code(f"...{ctx['context'][:200]}...")
                
                # Повторы
                if 'repetitiveness' in results and results['repetitiveness']['all_repetitions']:
                    with st.expander(f"🔄 Повторы ({len(results['repetitiveness']['all_repetitions'])} найдено)"):
                        for rep in results['repetitiveness']['all_repetitions'][:100]:
                            st.markdown(f"**{rep['ngram']}** — {rep['count']} раз(а)")
                
                # Тире
                if 'dashes' in results and results['dashes']['all_dash_sentences']:
                    with st.expander(f"— Тире ({len(results['dashes']['all_dash_sentences'])} найдено)"):
                        for item in results['dashes']['all_dash_sentences'][:50]:
                            st.info(f"Тире: {item['dash_count']} → {item['sentence'][:200]}")
                
                # Unicode-артефакты
                if 'unicode' in results and results['unicode']['all_suspicious_chunks']:
                    with st.expander(f"🔣 Unicode-артефакты ({len(results['unicode']['all_suspicious_chunks'])} найдено)"):
                        for chunk in results['unicode']['all_suspicious_chunks'][:100]:
                            st.code(f"Символ '{chunk['char']}' ({chunk['type']}) → ...{chunk['context'][:150]}...")
        
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Загрузите файл и нажмите 'Анализ'")
        
        # Компактная информация
        with st.expander("ℹ️ О модулях"):
            st.markdown("""
            **Новые модули 2025-2026:**
            - 📝 Длинные скобки (>5 слов) — признак человека
            - ‼️ Пунктуация ! ? ; — разнообразие = человек
            - 🔤 Апострофы 's — часто = ИИ
            - 🔢 Перечисления X, Y, and Z — часто = ИИ
            - 📊 Хеджинг — ключевой маркер (низкий = ИИ)
            - 🔄 Повторы 3-4-грамм — частые = ИИ
            - 📚 Лексическое разнообразие (MTLD/MATTR)
            - 🤖 ML-классификатор
            - 📉 Log-probability анализ
            """)


if __name__ == "__main__":
    main()
