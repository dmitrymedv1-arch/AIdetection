"""
Streamlit приложение для анализа научных статей на предмет использования ИИ
Версия 3.0 - Арт-галерея: Текст как искусство
Добавлены: полный анализ до References, новые маркеры, расширенная статистика
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
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = torch.cuda.is_available() or True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Для sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Конфигурация страницы
st.set_page_config(
    page_title="🎨 AI Detector | Текст как искусство",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Стилизация в стиле арт-галереи
st.markdown("""
<style>
    /* Основные стили галереи */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 20px rgba(102, 126, 234, 0.3);
    }
    
    .gallery-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .art-panel {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid;
        transition: transform 0.3s;
    }
    
    .art-panel:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .human-tone {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe0b2 100%);
        border-left-color: #ff6b6b;
    }
    
    .ai-tone {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left-color: #2196f3;
    }
    
    .canvas-text {
        font-family: 'Georgia', serif;
        line-height: 1.8;
        padding: 2rem;
        background: #fafafa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        position: relative;
        overflow: hidden;
    }
    
    .canvas-text::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.8) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .brush-stroke {
        background: linear-gradient(90deg, transparent, #ffd700, transparent);
        height: 2px;
        width: 100%;
        margin: 1rem 0;
        opacity: 0.5;
    }
    
    .frame-box {
        border: 3px solid #333;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 0 0 2px #fff, 0 0 0 5px #333;
        background: white;
    }
    
    .gallery-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-frame {
        background: white;
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .metric-frame:hover {
        border-color: #667eea;
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .color-temp-hot {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
    }
    
    .color-temp-cold {
        background: linear-gradient(45deg, #48dbfb, #5f27cd);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5253 100%);
        border-left: 5px solid #c62828;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        border-radius: 10px;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9f43 100%);
        border-left: 5px solid #f9a825;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        border-radius: 10px;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%);
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        border-radius: 10px;
    }
    
    .highlight-human {
        background-color: rgba(255, 107, 107, 0.2);
        border-bottom: 2px solid #ff6b6b;
        padding: 0.2rem;
        border-radius: 3px;
    }
    
    .highlight-ai {
        background-color: rgba(72, 219, 251, 0.2);
        border-bottom: 2px solid #48dbfb;
        padding: 0.2rem;
        border-radius: 3px;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #5f27cd);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Классы и функции для анализа
# ============================================================================

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
            # Расширенная статистика
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0,
                'percentiles': {}
            }
        }
        
        total_chars = len(text)
        if total_chars == 0:
            return results
        
        # Считаем слова для нормировки
        words = text.split()
        total_words = len(words)
        
        # Проходим по тексту (без ограничения)
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
        
        # Расширенная статистика
        char_types = [results['sup_sub_count'], results['fullwidth_count'], 
                     results['homoglyph_count'], results['greek_count'], 
                     results['cyrillic_count'], results['arabic_count']]
        
        results['stats']['mean'] = float(np.mean(char_types))
        results['stats']['median'] = float(np.median(char_types))
        results['stats']['max'] = float(np.max(char_types))
        results['stats']['std'] = float(np.std(char_types))
        results['stats']['percentiles'] = {
            '25': float(np.percentile(char_types, 25)),
            '75': float(np.percentile(char_types, 75))
        }
        
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
        results['suspicious_chunks'] = results['suspicious_chunks'][:5]
        
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
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0,
                'percentiles': {}
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
                if len(results['examples']) < 3:
                    results['examples'].append(sent[:150])
        
        # Статистика по тире
        if dash_counts:
            results['stats']['mean'] = float(np.mean(dash_counts))
            results['stats']['median'] = float(np.median(dash_counts))
            results['stats']['max'] = float(np.max(dash_counts))
            results['stats']['std'] = float(np.std(dash_counts))
            results['stats']['percentiles'] = {
                '25': float(np.percentile(dash_counts, 25)),
                '75': float(np.percentile(dash_counts, 75))
            }
        
        if sentences:
            results['percentage_heavy'] = (len(results['heavy_sentences']) / len(sentences)) * 100
        
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
        results['sentences_with_multiple_dashes'] = results['sentences_with_multiple_dashes'][:10]
        results['heavy_sentences'] = results['heavy_sentences'][:5]
        
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
            
            # ДОБАВЛЕНО: новые фразы по заданию
            'pathway', 'pathways',
            'signaling', 'signals',
            'Collectively',
            'manifest',
            'paradigm',  # уже есть, но оставляем для явности
            
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
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0,
                'percentiles': {}
            }
        }
        
        if not text or not sentences:
            return results
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        # Подсчет каждой фразы
        phrase_counts_list = []
        for phrase in self.ai_phrases:
            count = text_lower.count(phrase.lower())
            if count > 0:
                results['phrase_counts'][phrase] = count
                phrase_counts_list.append(count)
        
        # Статистика по фразам
        if phrase_counts_list:
            results['stats']['mean'] = float(np.mean(phrase_counts_list))
            results['stats']['median'] = float(np.median(phrase_counts_list))
            results['stats']['max'] = float(np.max(phrase_counts_list))
            results['stats']['std'] = float(np.std(phrase_counts_list))
            results['stats']['percentiles'] = {
                '25': float(np.percentile(phrase_counts_list, 25)),
                '75': float(np.percentile(phrase_counts_list, 75))
            }
        
        # Топ-10 самых частых фраз
        if results['phrase_counts']:
            sorted_phrases = sorted(results['phrase_counts'].items(), 
                                   key=lambda x: x[1], reverse=True)
            results['top_phrases'] = sorted_phrases[:10]
            
            # Отмечаем фразы, которые встречаются слишком часто
            for phrase, count in sorted_phrases:
                if count > 4:
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
        
        # Вероятностная оценка риска
        risk_score = 0
        confidence = 0.5
        
        if len(results['repeated_phrases']) > 2:
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
        
        if risk_score >= 3:
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
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'min': 0,
                'std': 0,
                'percentiles': {}
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
        
        # Расширенная статистика
        results['stats']['mean'] = float(np.mean(sent_lengths))
        results['stats']['median'] = float(np.median(sent_lengths))
        results['stats']['max'] = float(np.max(sent_lengths))
        results['stats']['min'] = float(np.min(sent_lengths))
        results['stats']['std'] = float(np.std(sent_lengths))
        results['stats']['percentiles'] = {
            '25': float(np.percentile(sent_lengths, 25)),
            '75': float(np.percentile(sent_lengths, 75))
        }
        
        # Коэффициент вариации (CV)
        if results['mean_length'] > 0:
            results['cv'] = float(results['std_length'] / results['mean_length'])
        
        # Межквартильный размах (IQR)
        q75, q25 = np.percentile(sent_lengths, [75, 25])
        results['iqr'] = float(q75 - q25)
        
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
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0,
                'percentiles': {}
            }
        }
        
        if not text:
            return results
        
        # Простая сегментация на предложения
        sentences = re.split(r'[.!?]+', text)
        results['total_sentences'] = len([s for s in sentences if len(s.strip()) > 10])
        
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
        
        # Статистика по грамматическим показателям
        grammar_metrics = [
            results['passive_indicators'],
            results['nominalization_count'],
            results['modal_count'],
            results['epistemic_count'],
            results['certainty_boosters_count']
        ]
        
        results['stats']['mean'] = float(np.mean(grammar_metrics))
        results['stats']['median'] = float(np.median(grammar_metrics))
        results['stats']['max'] = float(np.max(grammar_metrics))
        results['stats']['std'] = float(np.std(grammar_metrics))
        results['stats']['percentiles'] = {
            '25': float(np.percentile(grammar_metrics, 25)),
            '75': float(np.percentile(grammar_metrics, 75))
        }
        
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
        for match in passive_matches[:3]:
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
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0,
                'percentiles': {}
            }
        }
        
        if not text:
            return results
        
        text_lower = text.lower()
        words = text_lower.split()
        results['total_words'] = len(words)
        
        # Подсчет хеджинга
        hedging_counts = []
        for word in self.hedging_words:
            if ' ' in word:
                count = text_lower.count(word)
                results['hedging_count'] += count
                if count > 0:
                    hedging_counts.append(count)
            else:
                count = sum(1 for w in words if w == word)
                results['hedging_count'] += count
                if count > 0:
                    hedging_counts.append(count)
        
        # Подсчет категоричных выражений
        certainty_counts = []
        for phrase in self.certainty_phrases:
            if ' ' in phrase:
                count = text_lower.count(phrase)
                results['certainty_count'] += count
                if count > 0:
                    certainty_counts.append(count)
            else:
                count = sum(1 for w in words if w == phrase)
                results['certainty_count'] += count
                if count > 0:
                    certainty_counts.append(count)
        
        # Подсчет личных местоимений
        personal_counts = []
        for pronoun in self.personal_pronouns:
            count = sum(1 for w in words if w == pronoun)
            results['personal_count'] += count
            if count > 0:
                personal_counts.append(count)
        
        # Статистика
        all_counts = hedging_counts + certainty_counts + personal_counts
        if all_counts:
            results['stats']['mean'] = float(np.mean(all_counts))
            results['stats']['median'] = float(np.median(all_counts))
            results['stats']['max'] = float(np.max(all_counts))
            results['stats']['std'] = float(np.std(all_counts))
            results['stats']['percentiles'] = {
                '25': float(np.percentile(all_counts, 25)),
                '75': float(np.percentile(all_counts, 75))
            }
        
        # Нормировка на 1000 слов
        if results['total_words'] > 0:
            results['hedging_per_1000'] = (results['hedging_count'] * 1000) / results['total_words']
            results['certainty_per_1000'] = (results['certainty_count'] * 1000) / results['total_words']
            results['personal_per_1000'] = (results['personal_count'] * 1000) / results['total_words']
            
            if results['certainty_count'] > 0:
                results['hedging_ratio'] = results['hedging_count'] / results['certainty_count']
        
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
        
        if risk_score >= 4:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
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
            'intra_paragraph_similarity': 0,
            'inter_paragraph_similarity': 0,
            'transition_smoothness': 0,
            'structure_repetition': 0,
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'note': 'Paragraph analysis requires sentence-transformers',
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0,
                'percentiles': {}
            }
        }
        
        if not self.available or len(text) < 500:
            return results
        
        paragraphs = self.split_paragraphs(text)
        if len(paragraphs) < 3:
            return results
        
        results['paragraphs'] = paragraphs[:10]  # Ограничиваем
        
        # Статистика по длине абзацев
        para_lengths = [len(p.split()) for p in paragraphs]
        results['stats']['mean'] = float(np.mean(para_lengths))
        results['stats']['median'] = float(np.median(para_lengths))
        results['stats']['max'] = float(np.max(para_lengths))
        results['stats']['std'] = float(np.std(para_lengths))
        results['stats']['percentiles'] = {
            '25': float(np.percentile(para_lengths, 25)),
            '75': float(np.percentile(para_lengths, 75))
        }
        
        # Если нет модели, используем простые метрики
        if not self.model:
            # Простая метрика: вариативность длины абзацев
            if len(para_lengths) > 1:
                cv_lengths = np.std(para_lengths) / np.mean(para_lengths)
                if cv_lengths < 0.3:
                    results['risk_score'] = 2
                    results['confidence'] = 0.6
                    results['risk_level'] = 'medium'
                    results['note'] = 'Simple paragraph length analysis'
            return results
        
        # Полноценный анализ с эмбеддингами
        try:
            # Получаем эмбеддинги для абзацев
            para_embeddings = self.model.encode(paragraphs[:20])  # Ограничиваем
            
            # Внутриабзацная похожесть (разбиваем каждый абзац на предложения)
            intra_similarities = []
            for para in paragraphs[:5]:  # Первые 5 абзацев
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


class ParenthesisAnalyzer:
    """НОВЫЙ МОДУЛЬ: Анализ длинных пояснений в скобках"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str) -> Dict:
        """Анализирует содержание в круглых скобках"""
        results = {
            'total_parentheses': 0,
            'short_parentheses': 0,  # <5 слов
            'long_parentheses': 0,    # >=5 слов
            'long_parentheses_per_1000': 0,
            'examples_long': [],
            'examples_short': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'stats': {
                'mean_length': 0,
                'median_length': 0,
                'max_length': 0,
                'std_length': 0,
                'percentiles': {}
            }
        }
        
        if not text:
            return results
        
        # Находим все содержимое в круглых скобках
        parentheses_pattern = r'\(([^)]+)\)'
        matches = re.findall(parentheses_pattern, text)
        
        results['total_parentheses'] = len(matches)
        
        if not matches:
            return results
        
        # Анализируем длину каждого содержимого
        lengths = []
        for match in matches:
            word_count = len(match.split())
            lengths.append(word_count)
            
            if word_count >= 5:
                results['long_parentheses'] += 1
                if len(results['examples_long']) < 5:
                    results['examples_long'].append({
                        'text': match[:100] + '...' if len(match) > 100 else match,
                        'word_count': word_count
                    })
            else:
                results['short_parentheses'] += 1
                if len(results['examples_short']) < 5:
                    results['examples_short'].append({
                        'text': match[:100] + '...' if len(match) > 100 else match,
                        'word_count': word_count
                    })
        
        # Статистика по длине
        if lengths:
            results['stats']['mean_length'] = float(np.mean(lengths))
            results['stats']['median_length'] = float(np.median(lengths))
            results['stats']['max_length'] = float(np.max(lengths))
            results['stats']['std_length'] = float(np.std(lengths))
            results['stats']['percentiles'] = {
                '25': float(np.percentile(lengths, 25)),
                '75': float(np.percentile(lengths, 75))
            }
        
        # Нормировка на 1000 слов
        words = text.split()
        total_words = len(words)
        if total_words > 0:
            results['long_parentheses_per_1000'] = (results['long_parentheses'] * 1000) / total_words
        
        # Оценка риска (длинные пояснения в скобках - признак человека)
        risk_score = 0
        confidence = 0.5
        
        # Если есть длинные пояснения в скобках - это ХОРОШО (человек)
        if results['long_parentheses'] > 3:
            risk_score = 0  # Низкий риск
            confidence = 0.8
            results['risk_level'] = 'low'
        elif results['long_parentheses'] > 0:
            risk_score = 0
            confidence = 0.6
            results['risk_level'] = 'low'
        else:
            # Нет длинных пояснений - возможно ИИ
            if results['total_parentheses'] > 5:  # Если есть скобки, но все короткие
                risk_score = 2
                confidence = 0.7
                results['risk_level'] = 'medium'
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        return results


class PunctuationAnalyzer:
    """НОВЫЙ МОДУЛЬ: Анализ пунктуации ! ? ;"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует использование ! ? и ;"""
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
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0,
                'percentiles': {}
            }
        }
        
        if not text:
            return results
        
        # Подсчет знаков
        results['exclamation_count'] = text.count('!')
        results['question_count'] = text.count('?')
        results['semicolon_count'] = text.count(';')
        
        words = text.split()
        total_words = len(words)
        
        if total_words > 0:
            results['exclamation_per_1000'] = (results['exclamation_count'] * 1000) / total_words
            results['question_per_1000'] = (results['question_count'] * 1000) / total_words
            results['semicolon_per_1000'] = (results['semicolon_count'] * 1000) / total_words
        
        # Анализ контекста точки с запятой
        if results['semicolon_count'] > 0:
            # Ищем предложения с ;
            for sent in sentences[:20]:  # Ограничим для производительности
                if ';' in sent:
                    parts = sent.split(';')
                    if len(parts) >= 2:
                        # Проверяем, является ли вторая часть короткой
                        first_part = parts[0].strip()
                        second_part = parts[1].strip().split('.')[0]  # до точки
                        
                        if len(second_part.split()) < 5:  # короткое продолжение
                            results['semicolon_contexts'].append({
                                'full': sent[:100] + '...' if len(sent) > 100 else sent,
                                'first': first_part[:50] + '...' if len(first_part) > 50 else first_part,
                                'second': second_part[:50] + '...' if len(second_part) > 50 else second_part,
                                'type': 'short_continuation'
                            })
        
        # Статистика по пунктуации
        punct_counts = [
            results['exclamation_count'],
            results['question_count'],
            results['semicolon_count']
        ]
        
        results['stats']['mean'] = float(np.mean(punct_counts))
        results['stats']['median'] = float(np.median(punct_counts))
        results['stats']['max'] = float(np.max(punct_counts))
        results['stats']['std'] = float(np.std(punct_counts))
        results['stats']['percentiles'] = {
            '25': float(np.percentile(punct_counts, 25)),
            '75': float(np.percentile(punct_counts, 75))
        }
        
        # Оценка риска (наличие ! ? и особенно ; с коротким продолжением - признак человека)
        risk_score = 0
        confidence = 0.5
        
        # Восклицательные знаки - редкость в научных статьях, но если есть - человек
        if results['exclamation_count'] > 0:
            risk_score = max(0, risk_score - 1)  # Уменьшаем риск
        
        # Вопросительные знаки (риторические вопросы) - признак человека
        if results['question_count'] > 3:
            risk_score = max(0, risk_score - 1)
        
        # Точка с запятой с коротким продолжением - сильный признак человека
        if len(results['semicolon_contexts']) > 2:
            risk_score = max(0, risk_score - 2)
            confidence = 0.8
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        if risk_score == 0:
            results['risk_level'] = 'low'
        elif risk_score <= 1:
            results['risk_level'] = 'medium'
        else:
            results['risk_level'] = 'high'
        
        return results


class ApostropheAnalyzer:
    """НОВЫЙ МОДУЛЬ: Анализ апострофов ('s)"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str) -> Dict:
        """Анализирует использование апострофов в формате word's"""
        results = {
            'apostrophe_count': 0,
            'apostrophe_per_1000': 0,
            'examples': [],
            'unique_words': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'stats': {
                'mean_freq': 0,
                'median_freq': 0,
                'max_freq': 0,
                'std_freq': 0
            }
        }
        
        if not text:
            return results
        
        # Паттерн для поиска word's (буквы + апостроф + s)
        pattern = r'\b([a-zA-Z]+)\'s\b'
        matches = re.findall(pattern, text)
        
        results['apostrophe_count'] = len(matches)
        
        words = text.split()
        total_words = len(words)
        
        if total_words > 0:
            results['apostrophe_per_1000'] = (results['apostrophe_count'] * 1000) / total_words
        
        # Собираем примеры и уникальные слова
        word_counts = {}
        for match in matches:
            word = match.lower()
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Топ-10 слов с апострофом
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        results['unique_words'] = sorted_words[:10]
        
        # Примеры в контексте
        for match in matches[:5]:
            # Ищем контекст
            context_pattern = r'([^.]*?' + re.escape(match) + r'\'s[^.]*\.)'
            context_matches = re.findall(context_pattern, text[:5000])
            if context_matches:
                results['examples'].append({
                    'word': match,
                    'context': context_matches[0][:100] + '...' if len(context_matches[0]) > 100 else context_matches[0]
                })
        
        # Статистика по частоте
        if word_counts:
            frequencies = list(word_counts.values())
            results['stats']['mean_freq'] = float(np.mean(frequencies))
            results['stats']['median_freq'] = float(np.median(frequencies))
            results['stats']['max_freq'] = float(np.max(frequencies))
            results['stats']['std_freq'] = float(np.std(frequencies))
        
        # Оценка риска (частые апострофы - признак ИИ по заданию)
        risk_score = 0
        confidence = 0.5
        
        if results['apostrophe_per_1000'] > 2:
            risk_score = 3
            confidence = 0.9
        elif results['apostrophe_per_1000'] > 1:
            risk_score = 2
            confidence = 0.7
        elif results['apostrophe_per_1000'] > 0.5:
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


class EnumerationAnalyzer:
    """НОВЫЙ МОДУЛЬ: Анализ строгих перечислений X, Y, and Z"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """Анализирует конструкции вида X, Y, and Z"""
        results = {
            'enumeration_count': 0,
            'enumeration_per_1000': 0,
            'examples': [],
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'stats': {
                'mean_length': 0,
                'median_length': 0,
                'max_length': 0,
                'std_length': 0
            }
        }
        
        if not text or not sentences:
            return results
        
        # Паттерн для поиска X, Y, and Z
        # Ищем: слово, запятая, слово, запятая, and, слово
        pattern = r'\b(\w+),\s+(\w+),\s+and\s+(\w+)\b'
        
        # Также ищем вариант с пробелами: specifically X, Y, and Z
        specific_pattern = r'\b(specifically|including|such as|namely)\s+(\w+),\s+(\w+),\s+and\s+(\w+)\b'
        
        enumerations = []
        
        # Поиск в тексте (ограничим для производительности)
        for sent in sentences[:50]:
            matches = re.findall(pattern, sent)
            for match in matches:
                enumerations.append({
                    'items': list(match),
                    'full': sent[:150] + '...' if len(sent) > 150 else sent,
                    'type': 'simple'
                })
            
            specific_matches = re.findall(specific_pattern, sent)
            for match in specific_matches:
                intro = match[0]
                items = list(match[1:])
                enumerations.append({
                    'intro': intro,
                    'items': items,
                    'full': sent[:150] + '...' if len(sent) > 150 else sent,
                    'type': 'specific'
                })
        
        results['enumeration_count'] = len(enumerations)
        results['examples'] = enumerations[:10]
        
        words = text.split()
        total_words = len(words)
        
        if total_words > 0:
            results['enumeration_per_1000'] = (results['enumeration_count'] * 1000) / total_words
        
        # Статистика по длине элементов перечисления
        if enumerations:
            lengths = [len(' '.join(e['items'])) for e in enumerations]
            results['stats']['mean_length'] = float(np.mean(lengths))
            results['stats']['median_length'] = float(np.median(lengths))
            results['stats']['max_length'] = float(np.max(lengths))
            results['stats']['std_length'] = float(np.std(lengths))
        
        # Оценка риска (частые перечисления - признак ИИ по заданию)
        risk_score = 0
        confidence = 0.5
        
        if results['enumeration_per_1000'] > 0.5:
            risk_score = 3
            confidence = 0.9
        elif results['enumeration_per_1000'] > 0.2:
            risk_score = 2
            confidence = 0.7
        elif results['enumeration_count'] > 0:
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


class PerplexityAnalyzer:
    """Анализ перплексии (упрощенная версия без transformers)"""
    
    def __init__(self):
        self.available = TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
    
    def analyze(self, text: str) -> Dict:
        """Упрощенный анализ - возвращает заглушку если модель недоступна"""
        results = {
            'perplexities': [],
            'mean_perplexity': 0,
            'median_perplexity': 0,
            'perplexity_variance': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'note': 'Perplexity analysis requires transformers library',
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0
            }
        }
        
        if not self.available or len(text) < 100:
            results['note'] = 'Perplexity analysis unavailable or text too short'
            return results
        
        # Здесь можно добавить реальный анализ перплексии
        # но для Streamlit Cloud лучше использовать упрощенную версию
        results['note'] = 'Perplexity analysis disabled for performance'
        
        return results


class SemanticAnalyzer:
    """Анализ семантической близости (упрощенная версия)"""
    
    def __init__(self):
        self.available = SENTENCE_TRANSFORMERS_AVAILABLE
    
    def analyze(self, sentences: List[str]) -> Dict:
        """Упрощенный семантический анализ"""
        results = {
            'similarities': [],
            'mean_similarity': 0,
            'similarity_variance': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'note': 'Semantic analysis requires sentence-transformers',
            'stats': {
                'mean': 0,
                'median': 0,
                'max': 0,
                'std': 0
            }
        }
        
        if not self.available or len(sentences) < 5:
            return results
        
        # Упрощенная версия - используем длину предложений как прокси
        lengths = [len(s) for s in sentences[:50] if s]
        if len(lengths) > 1:
            # Вычисляем "гладкость" как обратную дисперсию длин
            std_length = np.std(lengths)
            mean_length = np.mean(lengths)
            if mean_length > 0:
                cv = std_length / mean_length
                
                results['stats']['mean'] = float(np.mean(lengths))
                results['stats']['median'] = float(np.median(lengths))
                results['stats']['max'] = float(np.max(lengths))
                results['stats']['std'] = float(std_length)
                
                # Интерпретируем низкую вариативность как "гладкий" текст
                if cv < 0.3:
                    results['mean_similarity'] = 0.8
                    results['risk_score'] = 3
                    results['confidence'] = 0.7
                    results['risk_level'] = 'high'
                elif cv < 0.5:
                    results['mean_similarity'] = 0.6
                    results['risk_score'] = 2
                    results['confidence'] = 0.6
                    results['risk_level'] = 'medium'
                else:
                    results['mean_similarity'] = 0.3
                    results['risk_score'] = 1
                    results['confidence'] = 0.5
                    results['risk_level'] = 'low'
        
        return results


class IntegratedRiskScorer:
    """Интегральная оценка риска на основе всех модулей"""
    
    def __init__(self):
        # Веса модулей (сумма = 100) - обновлено с новыми модулями
        self.weights = {
            'unicode': 0.05,        # 5% - почти исчез
            'dashes': 0.05,          # 5% - встречается
            'phrases': 0.12,         # 12% - упали, но не до нуля
            'burstiness': 0.08,      # 8% - остается полезным
            'grammar': 0.12,         # 12% - грамматика важна
            'hedging': 0.15,         # 15% - стойкий маркер
            'paragraph': 0.12,       # 12% - абзацный анализ
            'parenthesis': 0.08,     # 8% - НОВЫЙ модуль
            'punctuation': 0.07,      # 7% - НОВЫЙ модуль
            'apostrophe': 0.06,       # 6% - НОВЫЙ модуль
            'enumeration': 0.06,      # 6% - НОВЫЙ модуль
            'perplexity': 0.02,       # 2% - если доступно
            'semantic': 0.02          # 2% - если доступно
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
                    norm_score = data['risk_score'] / 3.0
                    
                    # Учитываем уверенность модуля
                    confidence = data.get('confidence', 0.5)
                    
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
        if final_score < 25:
            risk_level = 'low'
        elif final_score < 45:
            risk_level = 'medium-low'
        elif final_score < 65:
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
    def cut_at_references(text: str) -> str:
        """Обрезает текст до секции References/Bibliography"""
        if not text:
            return text
        
        # Маркеры начала списка литературы (разные варианты)
        reference_markers = [
            r'\nREFERENCES\s*\n',
            r'\nREFERENCE\s*\n',
            r'\nBIBLIOGRAPHY\s*\n',
            r'\nLITERATURE CITED\s*\n',
            r'\nWORKS CITED\s*\n',
            r'\nREFERENCES AND NOTES\s*\n',
            r'\nREFERENCES\n',
            r'\nBIBLIOGRAPHY\n',
            r'\nLITERATURE\n',
            r'\nCited References\n',
            r'\nReference List\n',
            r'\nReferences and Notes\n',
            r'\nREFERENCES\n',
            r'\nREFERENCES:\n',
            r'\nReferences\n',
            r'\nReferences:\n',
            r'\nBIBLIOGRAPHY\n',
            r'\nBibliography\n',
            r'\nBibliography:\n'
        ]
        
        # Ищем первый маркер и обрезаем текст до него
        text_lower = text.lower()
        for marker in reference_markers:
            # Убираем экранирование для поиска
            search_marker = marker.replace(r'\n', '\n').replace(r'\s*', '')
            pos = text.find(search_marker)
            if pos != -1:
                return text[:pos].strip()
        
        # Если не нашли, ищем по ключевым словам в начале строк
        lines = text.split('\n')
        cut_index = len(lines)
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(word in line_lower for word in ['references', 'bibliography', 'literature cited', 'works cited']):
                if len(line) < 50:  # Вероятно, заголовок
                    cut_index = i
                    break
        
        if cut_index < len(lines):
            return '\n'.join(lines[:cut_index]).strip()
        
        return text
    
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
            
            text = '\n'.join(full_text)
            # Обрезаем до References
            return DocumentProcessor.cut_at_references(text)
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
                    text = result.stdout
                    return DocumentProcessor.cut_at_references(text)
            except:
                pass
            
            # Если antiword не сработал, пробуем текстовый fallback
            os.unlink(tmp_path)
            
            # Пробуем прочитать как текстовый файл
            file.seek(0)
            content = file.getvalue().decode('utf-8', errors='ignore')
            if len(content) > 100:  # Хотя бы что-то получили
                return DocumentProcessor.cut_at_references(content)
            
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
    st.markdown('<h1 class="main-header">🎨 ТЕКСТ КАК ИСКУССТВО</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="gallery-container">
        <div style="text-align: center; font-size: 1.2rem; color: #555;">
            🖼️ Галерея текста: где тепло человеческого стиля встречается с холодом искусственного интеллекта
        </div>
        <div class="brush-stroke"></div>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
            <span class="color-temp-hot">🟨🟧🟫 Теплые тона: Человек</span>
            <span class="color-temp-cold">🟦🟪🟩 Холодные тона: ИИ</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "📤 Загрузите полотно (научную статью в .docx или .doc)", 
        type=['docx', 'doc'],
        help="Поддерживаются форматы .docx и .doc. Анализ выполняется до раздела References."
    )
    
    if uploaded_file is not None:
        # Показываем информацию о файле в стиле галереи
        file_details = {
            "Название произведения": uploaded_file.name,
            "Формат": uploaded_file.type or "application/octet-stream",
            "Размер полотна": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div class="art-panel human-tone">
                <h4>📋 Информация о произведении</h4>
            </div>
            """, unsafe_allow_html=True)
            st.json(file_details)
        with col2:
            if st.button("🎨 Начать анализ"):
                st.rerun()
        
        # Прогресс в стиле мазков кистью
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Шаг 1: Чтение документа
            status_text.text("🎨 Чтение полотна...")
            progress_bar.progress(5)
            
            if uploaded_file.name.endswith('.docx'):
                text = DocumentProcessor.read_docx(uploaded_file)
            else:  # .doc
                text = DocumentProcessor.read_doc(uploaded_file)
            
            if not text or len(text.strip()) < 100:
                st.warning("⚠️ Полотно слишком мало или не содержит текста")
                return
            
            # Шаг 2: Предобработка
            status_text.text("🖌️ Подготовка холста...")
            progress_bar.progress(10)
            text = DocumentProcessor.preprocess(text)
            
            # Шаг 3: Сегментация на предложения
            status_text.text("🔍 Анализ мазков (предложений)...")
            progress_bar.progress(15)
            sentences = DocumentProcessor.split_sentences_simple(text)
            
            st.success(f"✅ Полотно загружено: {len(text)} символов, {len(sentences)} мазков (предложений)")
            
            # Создаем вкладки для галереи
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🖼️ ГАЛЕРЕЯ", 
                "🔍 ДЕТАЛИ АРТЕФАКТОВ", 
                "📊 СТАТИСТИКА",
                "🎨 ИНТЕГРАЛЬНАЯ ОЦЕНКА",
                "📝 ПОЛОТНО"
            ])
            
            # Хранилище результатов (все модули активны)
            results = {}
            risk_scores = []
            
            # =================================================================
            # Модуль 1: Unicode-артефакты
            # =================================================================
            with st.spinner("Анализ Unicode-артефактов и нелатинских символов..."):
                detector = UnicodeArtifactDetector()
                results['unicode'] = detector.analyze(text)
                risk_scores.append(results['unicode']['risk_score'])
            progress_bar.progress(20)
            
            # =================================================================
            # Модуль 2: Множественные тире
            # =================================================================
            with st.spinner("Анализ множественных тире..."):
                detector = DashAnalyzer()
                results['dashes'] = detector.analyze(sentences)
                risk_scores.append(results['dashes']['risk_score'])
            progress_bar.progress(25)
            
            # =================================================================
            # Модуль 3: ИИ-фразы
            # =================================================================
            with st.spinner("Поиск ИИ-фраз (обновленный список 2025-2026)..."):
                detector = AIPhraseDetector()
                results['phrases'] = detector.analyze(text, sentences)
                risk_scores.append(results['phrases']['risk_score'])
            progress_bar.progress(30)
            
            # =================================================================
            # Модуль 4: Burstiness
            # =================================================================
            with st.spinner("Анализ вариативности предложений..."):
                detector = BurstinessAnalyzer()
                results['burstiness'] = detector.analyze(sentences)
                if 'error' not in results['burstiness']:
                    risk_scores.append(results['burstiness']['risk_score'])
            progress_bar.progress(35)
            
            # =================================================================
            # Модуль 5: Грамматика
            # =================================================================
            with st.spinner("Грамматический анализ (модальность, номинализации)..."):
                detector = GrammarAnalyzer()
                results['grammar'] = detector.analyze(text)
                risk_scores.append(results['grammar']['risk_score'])
            progress_bar.progress(40)
            
            # =================================================================
            # Модуль 6: Хеджинг
            # =================================================================
            with st.spinner("Анализ хеджинга (ключевой маркер)..."):
                detector = HedgingAnalyzer()
                results['hedging'] = detector.analyze(text)
                risk_scores.append(results['hedging']['risk_score'])
            progress_bar.progress(45)
            
            # =================================================================
            # Модуль 7: Абзацный анализ
            # =================================================================
            with st.spinner("Анализ на уровне абзацев..."):
                detector = ParagraphAnalyzer()
                results['paragraph'] = detector.analyze(text, sentences)
                if 'error' not in results['paragraph']:
                    risk_scores.append(results['paragraph']['risk_score'])
            progress_bar.progress(50)
            
            # =================================================================
            # Модуль 8: Parenthesis (НОВЫЙ) - анализ скобок
            # =================================================================
            with st.spinner("Анализ пояснений в скобках..."):
                detector = ParenthesisAnalyzer()
                results['parenthesis'] = detector.analyze(text)
                risk_scores.append(results['parenthesis']['risk_score'])
            progress_bar.progress(55)
            
            # =================================================================
            # Модуль 9: Punctuation (НОВЫЙ) - анализ ! ? ;
            # =================================================================
            with st.spinner("Анализ пунктуации (! ? ;)..."):
                detector = PunctuationAnalyzer()
                results['punctuation'] = detector.analyze(text, sentences)
                risk_scores.append(results['punctuation']['risk_score'])
            progress_bar.progress(60)
            
            # =================================================================
            # Модуль 10: Apostrophe (НОВЫЙ) - анализ 's
            # =================================================================
            with st.spinner("Анализ апострофов ('s)..."):
                detector = ApostropheAnalyzer()
                results['apostrophe'] = detector.analyze(text)
                risk_scores.append(results['apostrophe']['risk_score'])
            progress_bar.progress(65)
            
            # =================================================================
            # Модуль 11: Enumeration (НОВЫЙ) - анализ X, Y, and Z
            # =================================================================
            with st.spinner("Анализ перечислений (X, Y, and Z)..."):
                detector = EnumerationAnalyzer()
                results['enumeration'] = detector.analyze(text, sentences)
                risk_scores.append(results['enumeration']['risk_score'])
            progress_bar.progress(70)
            
            # =================================================================
            # Модуль 12: Perplexity
            # =================================================================
            with st.spinner("Анализ перплексии..."):
                detector = PerplexityAnalyzer()
                results['perplexity'] = detector.analyze(text)
                if 'error' not in results['perplexity']:
                    risk_scores.append(results['perplexity']['risk_score'])
            progress_bar.progress(75)
            
            # =================================================================
            # Модуль 13: Semantic
            # =================================================================
            with st.spinner("Семантический анализ..."):
                detector = SemanticAnalyzer()
                results['semantic'] = detector.analyze(sentences)
                if 'error' not in results['semantic']:
                    risk_scores.append(results['semantic']['risk_score'])
            progress_bar.progress(80)
            
            # Интегральная оценка
            scorer = IntegratedRiskScorer()
            integrated_results = scorer.calculate(results)
            
            # Завершение
            progress_bar.progress(100)
            status_text.text("✅ Анализ завершен! Полотно готово к просмотру.")
            
            # =================================================================
            # Вкладка 1: ГАЛЕРЕЯ (общий отчет в стиле арт-галереи)
            # =================================================================
            with tab1:
                st.markdown("""
                <div class="frame-box">
                    <h2 style="text-align: center;">🖼️ ГАЛЕРЕЯ ТЕКСТА</h2>
                    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
                        Теплые тона - человеческий стиль | Холодные тона - стиль ИИ
                    </div>
                """, unsafe_allow_html=True)
                
                # Интегральный риск в виде цветовой температуры
                final_score = integrated_results['final_score']
                risk_level = integrated_results['risk_level']
                
                if final_score < 25:
                    temp_class = "risk-low"
                    temp_text = "🟢 ТЕПЛЫЕ ТОНА: Высокая вероятность человеческого текста"
                    temp_color = "#1dd1a1"
                elif final_score < 45:
                    temp_class = "risk-low"
                    temp_text = "🟡 ТЕПЛЫЕ ТОНА С ПРИМЕСЬЮ: Преимущественно человек"
                    temp_color = "#feca57"
                elif final_score < 65:
                    temp_class = "risk-medium"
                    temp_text = "🟠 СМЕШАННАЯ ТЕХНИКА: Средняя вероятность ИИ"
                    temp_color = "#ff9f43"
                else:
                    temp_class = "risk-high"
                    temp_text = "🔴 ХОЛОДНЫЕ ТОНА: Высокая вероятность ИИ"
                    temp_color = "#ff6b6b"
                
                st.markdown(f"""
                <div class="{temp_class}" style="text-align: center; font-size: 1.5rem; margin: 2rem 0;">
                    {temp_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Метрики в стиле галереи
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="metric-frame">
                        <h3>🎨 Оценка</h3>
                    """, unsafe_allow_html=True)
                    st.metric("Интегральная", f"{final_score:.1f}/100")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-frame">
                        <h3>🖌️ Мазков</h3>
                    """, unsafe_allow_html=True)
                    st.metric("Предложений", len(sentences))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-frame">
                        <h3>📊 Активных</h3>
                    """, unsafe_allow_html=True)
                    st.metric("Модулей", len(risk_scores))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown("""
                    <div class="metric-frame">
                        <h3>🎯 Уверенность</h3>
                    """, unsafe_allow_html=True)
                    st.metric("Анализа", f"{integrated_results['total_confidence']:.0%}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Сводка по метрикам в стиле галереи
                st.markdown("""
                <div class="gallery-metrics">
                    <h3>🖼️ ЭКСПОЗИЦИЯ МЕТРИК</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Создаем цветные карточки для каждой метрики
                metric_cols = st.columns(3)
                metric_items = []
                
                for key, data in results.items():
                    if isinstance(data, dict):
                        risk_level = data.get('risk_level', 'unknown')
                        risk_score = data.get('risk_score', 0)
                        
                        # Определяем цвет на основе риска
                        if risk_level == 'high':
                            color_class = "ai-tone"
                            icon = "🔴"
                        elif risk_level == 'medium' or risk_level == 'medium-high' or risk_level == 'medium-low':
                            color_class = "human-tone"
                            icon = "🟡"
                        else:
                            color_class = "human-tone"
                            icon = "🟢"
                        
                        # Получаем основное значение
                        main_value = 'N/A'
                        if 'density_per_10k' in data:
                            main_value = f"{data['density_per_10k']:.2f}/10k"
                        elif 'percentage_heavy' in data:
                            main_value = f"{data['percentage_heavy']:.1f}%"
                        elif 'cv' in data:
                            main_value = f"{data['cv']:.3f}"
                        elif 'hedging_per_1000' in data:
                            main_value = f"{data['hedging_per_1000']:.1f}/1000"
                        elif 'passive_percentage' in data:
                            main_value = f"{data['passive_percentage']:.1f}%"
                        elif 'non_latin_per_1000' in data:
                            main_value = f"{data['non_latin_per_1000']:.2f}/1000"
                        elif 'long_parentheses' in data:
                            main_value = f"{data['long_parentheses']} длинных"
                        elif 'apostrophe_per_1000' in data:
                            main_value = f"{data['apostrophe_per_1000']:.2f}/1000"
                        elif 'enumeration_count' in data:
                            main_value = f"{data['enumeration_count']} перечислений"
                        else:
                            main_value = f"{risk_score}/3"
                        
                        metric_items.append({
                            'name': key.capitalize(),
                            'icon': icon,
                            'color': color_class,
                            'value': main_value,
                            'risk': risk_level.upper()
                        })
                
                # Отображаем метрики сеткой
                for i in range(0, len(metric_items), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(metric_items):
                            item = metric_items[i + j]
                            with cols[j]:
                                st.markdown(f"""
                                <div class="art-panel {item['color']}">
                                    <h4>{item['icon']} {item['name']}</h4>
                                    <p style="font-size: 1.5rem; font-weight: bold;">{item['value']}</p>
                                    <p style="color: #666;">{item['risk']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # =================================================================
            # Вкладка 2: ДЕТАЛИ АРТЕФАКТОВ
            # =================================================================
            with tab2:
                st.header("🔍 ДЕТАЛИ АРТЕФАКТОВ")
                
                # Создаем аккордеоны для каждого модуля
                if 'unicode' in results:
                    with st.expander("🎨 Unicode-артефакты и нелатинские символы", expanded=False):
                        u = results['unicode']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Плотность на 10k", f"{u['density_per_10k']:.2f}")
                            st.metric("Надстрочные/подстрочные", u['sup_sub_count'])
                            st.metric("Греческие символы", u['greek_count'])
                            st.metric("Среднее (стат)", f"{u['stats']['mean']:.2f}")
                        with col2:
                            st.metric("Fullwidth цифры", u['fullwidth_count'])
                            st.metric("Гомоглифы", u['homoglyph_count'])
                            st.metric("Кириллица", u['cyrillic_count'])
                            st.metric("Медиана (стат)", f"{u['stats']['median']:.2f}")
                        
                        if u['suspicious_chunks']:
                            st.write("**Примеры подозрительных символов:**")
                            for chunk in u['suspicious_chunks'][:3]:
                                st.code(f"Символ '{chunk['char']}' в контексте: ...{chunk['context']}...")
                
                if 'dashes' in results:
                    with st.expander("📏 Множественные тире", expanded=False):
                        d = results['dashes']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Предложений с ≥2 тире", len(d['sentences_with_multiple_dashes']))
                            st.metric("Тяжелые предложения", len(d['heavy_sentences']))
                        with col2:
                            st.metric("Процент тяжелых", f"{d['percentage_heavy']:.2f}%")
                            st.metric("Среднее тире/предл", f"{d['stats']['mean']:.2f}")
                        
                        if d['examples']:
                            st.write("**Примеры:**")
                            for i, ex in enumerate(d['examples'][:2]):
                                st.info(f"Пример {i+1}: {ex}")
                
                if 'phrases' in results:
                    with st.expander("🤖 ИИ-фразы и штампы", expanded=False):
                        p = results['phrases']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Плотность переходов", f"{p['transition_density']:.2f}/1000")
                            st.metric("Метадискурс на 1000", f"{p.get('metadiscourse_per_1000', 0):.2f}")
                        with col2:
                            st.metric("Повторяющихся фраз", len(p['repeated_phrases']))
                            st.metric("Уверенность", f"{p.get('confidence', 0.5):.0%}")
                        with col3:
                            st.metric("Среднее (стат)", f"{p['stats']['mean']:.2f}")
                            st.metric("Максимум (стат)", f"{p['stats']['max']:.2f}")
                        
                        if p['top_phrases']:
                            st.write("**Наиболее частые фразы:**")
                            for phrase, count in p['top_phrases'][:5]:
                                st.write(f"- '{phrase}': {count} раз(а)")
                        
                        if p['repeated_phrases']:
                            st.write("**⚠️ Часто повторяющиеся фразы:**")
                            for item in p['repeated_phrases']:
                                st.write(f"- '{item['phrase']}' ({item['count']} раз)")
                
                if 'parenthesis' in results:
                    with st.expander("📝 Анализ скобок ()", expanded=False):
                        par = results['parenthesis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Всего скобок", par['total_parentheses'])
                            st.metric("Коротких (<5 слов)", par['short_parentheses'])
                        with col2:
                            st.metric("Длинных (≥5 слов)", par['long_parentheses'])
                            st.metric("Длинных на 1000 слов", f"{par['long_parentheses_per_1000']:.2f}")
                        
                        st.metric("Средняя длина в скобках", f"{par['stats']['mean_length']:.1f} слов")
                        
                        if par['examples_long']:
                            st.write("**✨ Длинные пояснения (признак человека):**")
                            for ex in par['examples_long'][:3]:
                                st.success(f"📌 {ex['word_count']} слов: {ex['text']}")
                        
                        if par['examples_short']:
                            with st.expander("Короткие скобки (типично для ИИ)"):
                                for ex in par['examples_short'][:3]:
                                    st.info(f"🔹 {ex['word_count']} слов: {ex['text']}")
                
                if 'punctuation' in results:
                    with st.expander("❗ Анализ пунктуации (! ? ;)", expanded=False):
                        punct = results['punctuation']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Восклицательных знаков", punct['exclamation_count'])
                            st.metric("На 1000 слов", f"{punct['exclamation_per_1000']:.2f}")
                        with col2:
                            st.metric("Вопросительных знаков", punct['question_count'])
                            st.metric("На 1000 слов", f"{punct['question_per_1000']:.2f}")
                        with col3:
                            st.metric("Точек с запятой", punct['semicolon_count'])
                            st.metric("На 1000 слов", f"{punct['semicolon_per_1000']:.2f}")
                        
                        if punct['semicolon_contexts']:
                            st.write("**✨ Точка с запятой с коротким продолжением (признак человека):**")
                            for ctx in punct['semicolon_contexts'][:3]:
                                st.info(f"📌 {ctx['first']} → **; {ctx['second']}**")
                
                if 'apostrophe' in results:
                    with st.expander("🔤 Анализ апострофов ('s)", expanded=False):
                        ap = results['apostrophe']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Всего апострофов", ap['apostrophe_count'])
                            st.metric("На 1000 слов", f"{ap['apostrophe_per_1000']:.2f}")
                        with col2:
                            st.metric("Уникальных слов", len(ap['unique_words']))
                            st.metric("Средняя частота", f"{ap['stats']['mean_freq']:.2f}")
                        
                        if ap['unique_words']:
                            st.write("**📊 Слова с апострофом:**")
                            for word, count in ap['unique_words'][:5]:
                                st.write(f"- {word}'s: {count} раз")
                        
                        if ap['examples']:
                            st.write("**📝 Примеры в контексте:**")
                            for ex in ap['examples'][:3]:
                                st.info(f"📌 {ex['context']}")
                
                if 'enumeration' in results:
                    with st.expander("🔢 Анализ перечислений (X, Y, and Z)", expanded=False):
                        enum = results['enumeration']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Всего перечислений", enum['enumeration_count'])
                            st.metric("На 1000 слов", f"{enum['enumeration_per_1000']:.2f}")
                        with col2:
                            st.metric("Средняя длина", f"{enum['stats']['mean_length']:.1f} символов")
                            st.metric("Макс длина", f"{enum['stats']['max_length']:.1f}")
                        
                        if enum['examples']:
                            st.write("**📋 Примеры перечислений (признак ИИ):**")
                            for ex in enum['examples'][:5]:
                                if ex['type'] == 'specific':
                                    st.warning(f"📌 {ex['intro']}: {', '.join(ex['items'])}")
                                else:
                                    st.warning(f"📌 {', '.join(ex['items'])}")
            
            # =================================================================
            # Вкладка 3: СТАТИСТИКА
            # =================================================================
            with tab3:
                st.header("📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    if 'burstiness' in results and 'error' not in results['burstiness']:
                        st.subheader("📈 Burstiness (вариативность)")
                        b = results['burstiness']
                        
                        st.metric("Ср. длина предложения", f"{b['mean_length']:.1f} слов")
                        st.metric("Станд. отклонение", f"{b['std_length']:.1f}")
                        st.metric("Коэф. вариации", f"{b['cv']:.3f}")
                        st.metric("Медиана длины", f"{b['stats']['median']:.1f}")
                        st.metric("Мин-Макс", f"{b['stats']['min']:.0f} - {b['stats']['max']:.0f}")
                        
                        # Простая гистограмма
                        if b['sentence_lengths']:
                            fig = go.Figure(data=[go.Histogram(x=b['sentence_lengths'], nbinsx=20)])
                            fig.update_layout(
                                title="Распределение длины предложений",
                                xaxis_title="Длина (слова)",
                                yaxis_title="Частота",
                                height=300,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if 'grammar' in results:
                        st.subheader("📚 Грамматика")
                        g = results['grammar']
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Пассивных конструкций", f"{g['passive_percentage']:.1f}%")
                            st.metric("Модальные глаголы на 1000", f"{g.get('modals_per_1000', 0):.2f}")
                        with col_b:
                            st.metric("Номинализаций на 1000", f"{g['nominalizations_per_1000']:.1f}")
                            st.metric("Epistemic markers", f"{g.get('epistemic_per_1000', 0):.2f}")
                        
                        if g['examples']:
                            st.write("**Пример пассива:**")
                            for ex in g['examples'][:2]:
                                st.info(ex)
                
                with col_right:
                    if 'hedging' in results:
                        st.subheader("🤔 Хеджинг")
                        h = results['hedging']
                        
                        col_c, col_d = st.columns(2)
                        with col_c:
                            st.metric("Хеджинг на 1000", f"{h['hedging_per_1000']:.2f}")
                            st.metric("Категоричность на 1000", f"{h.get('certainty_per_1000', 0):.2f}")
                        with col_d:
                            st.metric("Личные местоимения", f"{h['personal_per_1000']:.2f}")
                            st.metric("Соотношение H/C", f"{h.get('hedging_ratio', 0):.2f}")
                        
                        # Индикатор
                        if h['hedging_per_1000'] < 3:
                            st.warning("⚠️ Очень низкий хеджинг - сильный сигнал ИИ")
                        elif h['hedging_per_1000'] < 5:
                            st.info("Умеренный уровень хеджинга")
                        else:
                            st.success("✅ Хороший уровень хеджинга")
                    
                    if 'paragraph' in results and 'error' not in results['paragraph']:
                        st.subheader("📑 Абзацы")
                        p = results['paragraph']
                        
                        if 'intra_paragraph_similarity' in p:
                            st.metric("Внутриабзацная похожесть", f"{p['intra_paragraph_similarity']:.3f}")
                            st.metric("Плавность переходов", f"{p.get('transition_smoothness', 0):.3f}")
                            st.metric("Ср. длина абзаца", f"{p['stats']['mean']:.1f} слов")
                            st.metric("Медиана длины", f"{p['stats']['median']:.1f} слов")
                            
                            if p.get('intra_paragraph_similarity', 0) > 0.72:
                                st.warning("⚠️ Очень высокая внутриабзацная похожесть")
                        
                        st.info(p.get('note', ''))
                    
                    if 'perplexity' in results:
                        st.subheader("📊 Perplexity")
                        pp = results['perplexity']
                        st.info(pp.get('note', 'Недоступно'))
            
            # =================================================================
            # Вкладка 4: ИНТЕГРАЛЬНАЯ ОЦЕНКА
            # =================================================================
            with tab4:
                st.markdown("""
                <div class="frame-box">
                    <h2 style="text-align: center;">🎨 ИНТЕГРАЛЬНАЯ ОЦЕНКА</h2>
                """, unsafe_allow_html=True)
                
                # Отображаем итоговый результат
                final_score = integrated_results['final_score']
                risk_level = integrated_results['risk_level']
                
                if risk_level == 'low':
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3>🟢 ТЕПЛЫЕ ТОНА: Низкий риск использования ИИ ({final_score:.1f}/100)</h3>
                        <p>Текст демонстрирует характеристики, типичные для человеческого письма:
                        хорошая вариативность, умеренный хеджинг, естественные переходы, 
                        наличие длинных пояснений в скобках и разнообразной пунктуации.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == 'medium-low':
                    st.markdown(f"""
                    <div class="risk-medium">
                        <h3>🟡 ТЕПЛЫЕ ТОНА С ПРИМЕСЬЮ: Умеренно-низкий риск ({final_score:.1f}/100)</h3>
                        <p>Текст имеет некоторые признаки ИИ-генерации, но в целом похож на человеческий.
                        Рекомендуется дополнительная проверка подозрительных секций.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == 'medium-high':
                    st.markdown(f"""
                    <div class="risk-medium">
                        <h3>🟠 СМЕШАННАЯ ТЕХНИКА: Умеренно-высокий риск ({final_score:.1f}/100)</h3>
                        <p>Текст демонстрирует множественные признаки ИИ-генерации:
                        низкий хеджинг, высокая внутриабзацная похожесть, повторяющиеся фразы,
                        частые перечисления X, Y, and Z, много апострофов.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3>🔴 ХОЛОДНЫЕ ТОНА: Высокий риск использования ИИ ({final_score:.1f}/100)</h3>
                        <p>Текст с высокой вероятностью сгенерирован ИИ. Обнаружены сильные маркеры:
                        очень низкий хеджинг, высокая однородность, характерные ИИ-фразы,
                        частые апострофы и строгие перечисления.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Диаграмма вклада модулей
                st.subheader("🎨 Вклад модулей в итоговую оценку")
                
                if integrated_results['module_scores']:
                    module_df = pd.DataFrame(integrated_results['module_scores'])
                    
                    # Сортируем по вкладу
                    module_df = module_df.sort_values('contribution', ascending=True)
                    
                    # Создаем цветовую шкалу от теплого к холодному
                    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#5f27cd']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=module_df['contribution'],
                            y=module_df['module'],
                            orientation='h',
                            marker=dict(
                                color=module_df['contribution'],
                                colorscale=[[0, '#ff6b6b'], [0.33, '#feca57'], 
                                           [0.66, '#48dbfb'], [1, '#5f27cd']],
                                showscale=True,
                                colorbar=dict(title="Вклад (%)")
                            ),
                            text=module_df['contribution'].round(1),
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Вклад модулей в итоговый риск (%)",
                        xaxis_title="Вклад в итоговую оценку (%)",
                        yaxis_title="Модуль",
                        height=500,
                        showlegend=False,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Таблица с деталями
                    st.subheader("📋 Детализация по модулям")
                    module_df['raw_score_display'] = module_df['raw_score'].apply(lambda x: f"{x}/3")
                    module_df['confidence_display'] = module_df['confidence'].apply(lambda x: f"{x:.0%}")
                    module_df['weight_display'] = module_df['weight'].apply(lambda x: f"{x:.1%}")
                    
                    display_df = module_df[['module', 'raw_score_display', 'confidence_display', 
                                           'weight_display', 'contribution']]
                    display_df.columns = ['Модуль', 'Оценка', 'Уверенность', 'Вес', 'Вклад (%)']
                    
                    # Цветовая кодировка для модулей
                    def color_rows(row):
                        if 'hedging' in row['Модуль']:
                            return ['background-color: #fff3cd'] * len(row)
                        elif 'apostrophe' in row['Модуль'] or 'enumeration' in row['Модуль']:
                            return ['background-color: #e3f2fd'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = display_df.style.apply(color_rows, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # =================================================================
            # Вкладка 5: ПОЛОТНО (текст с подсветкой)
            # =================================================================
            with tab5:
                st.markdown("""
                <div class="frame-box">
                    <h2 style="text-align: center;">📝 ПОЛОТНО ТЕКСТА</h2>
                    <p style="text-align: center; color: #666;">
                        🟨 Теплый фон - человеческие маркеры | 🟦 Холодный фон - маркеры ИИ
                    </p>
                """, unsafe_allow_html=True)
                
                # Показываем текст с подсветкой
                st.subheader("Текст статьи (полный текст, обрезан до References)")
                
                # Создаем подсвеченную версию текста
                highlighted_text = text
                
                # Подсвечиваем длинные пояснения в скобках (человек)
                if 'parenthesis' in results and results['parenthesis']['examples_long']:
                    for ex in results['parenthesis']['examples_long']:
                        if ex['text'] in highlighted_text:
                            highlighted_text = highlighted_text.replace(
                                ex['text'], 
                                f"<span class='highlight-human'>{ex['text']}</span>"
                            )
                
                # Подсвечиваем перечисления X, Y, and Z (ИИ)
                if 'enumeration' in results and results['enumeration']['examples']:
                    for ex in results['enumeration']['examples']:
                        if 'full' in ex:
                            full_text = ex['full'].replace('...', '')
                            if full_text in highlighted_text:
                                highlighted_text = highlighted_text.replace(
                                    full_text,
                                    f"<span class='highlight-ai'>{full_text}</span>"
                                )
                
                # Подсвечиваем апострофы (ИИ)
                if 'apostrophe' in results and results['apostrophe']['examples']:
                    for ex in results['apostrophe']['examples']:
                        if ex['word'] in highlighted_text:
                            highlighted_text = highlighted_text.replace(
                                ex['word'] + "'s",
                                f"<span class='highlight-ai'>{ex['word']}'s</span>"
                            )
                
                # Отображаем текст
                st.markdown(f"""
                <div class="canvas-text">
                    {highlighted_text[:10000]}{'...' if len(highlighted_text) > 10000 else ''}
                </div>
                """, unsafe_allow_html=True)
                
                if len(highlighted_text) > 10000:
                    st.info(f"📌 Показано 10000 из {len(highlighted_text)} символов. Полный текст доступен в скачанном файле.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Ошибка при обработке: {str(e)}")
            st.exception(e)
    
    else:
        # Информация о приложении в стиле галереи
        st.markdown("""
        <div class="gallery-container">
            <div style="text-align: center; padding: 3rem;">
                <h2>👆 Загрузите полотно для начала анализа</h2>
                <div class="brush-stroke" style="width: 50%; margin: 2rem auto;"></div>
                <p style="color: #666; font-size: 1.1rem;">
                    Научная статья станет произведением искусства, где каждый мазок (предложение) 
                    будет проанализирован и отнесен к теплым (человеческим) или холодным (ИИ) тонам.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ О ГАЛЕРЕЕ (метрики анализа 2025-2026)"):
            st.markdown("""
            ### 🎨 ЭКСПОЗИЦИЯ МЕТРИК:
            
            **Уровень 1: Артефакты**
            - **Unicode-артефакты**: надстрочные/подстрочные символы, fullwidth цифры
            - **Нелатинские символы**: греческие, кириллица, арабские
            - **Множественные тире**: ≥3 длинных тире в предложении
            - **ИИ-фразы 2025-2026**: pathway, signaling, Collectively, manifest, paradigm
            
            **Уровень 2: Статистика**
            - **Burstiness**: вариативность длины предложений
            - **Грамматика**: пассив, номинализации, модальные глаголы
            - **Хеджинг**: ключевой маркер - слова неуверенности
            
            **Уровень 3: Новые модули**
            - **Скобки ()**: длинные пояснения (≥5 слов) - признак человека
            - **Пунктуация ! ? ;**: разнообразие - признак человека
            - **Апострофы 's**: частые апострофы - признак ИИ
            - **Перечисления X, Y, and Z**: строгие перечисления - признак ИИ
            
            **Интегральная оценка:**
            - Теплые тона 🟨🟧🟫 → Человек
            - Холодные тона 🟦🟪🟩 → ИИ
            """)


if __name__ == "__main__":
    main()
