"""
Streamlit приложение для анализа научных статей на предмет использования ИИ
Исправленная версия с учетом совместимости версий и оптимизацией для Streamlit Cloud
Версия 3.0 - Добавлены: анализ скобок, пунктуации, апострофов, перечислений, полная статистика
Дизайн: Спортивная аналитика 2026
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
    page_title="🏆 ЧЕМПИОНАТ AI-ДЕТЕКЦИИ 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация под спортивную аналитику
st.markdown("""
<style>
    /* Основные стили */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Табло как на стадионе */
    .scoreboard {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 3px solid #ffd700;
        color: white;
        text-align: center;
    }
    
    .scoreboard-title {
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #ffd700;
        margin-bottom: 1rem;
    }
    
    .scoreboard-number {
        font-size: 5rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 20px #ffd700;
        line-height: 1;
    }
    
    .scoreboard-label {
        font-size: 1.5rem;
        text-transform: uppercase;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    /* Карточки игроков (модулей) */
    .player-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #1E88E5;
        transition: transform 0.3s;
        text-align: center;
    }
    
    .player-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .player-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1a237e;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .player-score {
        font-size: 2rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    
    .player-medal {
        font-size: 1.5rem;
        margin-top: 0.5rem;
    }
    
    /* Медали для уровней риска */
    .medal-gold {
        color: #ffd700;
        text-shadow: 0 0 10px #ffd700;
    }
    
    .medal-silver {
        color: #c0c0c0;
        text-shadow: 0 0 10px #c0c0c0;
    }
    
    .medal-bronze {
        color: #cd7f32;
        text-shadow: 0 0 10px #cd7f32;
    }
    
    /* Турнирная таблица */
    .tournament-table {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .tournament-row {
        display: flex;
        align-items: center;
        padding: 0.8rem;
        border-bottom: 1px solid #eee;
    }
    
    .tournament-rank {
        width: 40px;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .tournament-player {
        flex: 2;
        font-weight: 500;
    }
    
    .tournament-bar {
        flex: 3;
        height: 20px;
        background: #f0f2f6;
        border-radius: 10px;
        overflow: hidden;
        margin: 0 1rem;
    }
    
    .tournament-progress {
        height: 100%;
        background: linear-gradient(90deg, #4caf50, #ffc107, #f44336);
        border-radius: 10px;
        transition: width 0.5s;
    }
    
    .tournament-value {
        width: 70px;
        text-align: right;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    /* MVP карточка */
    .mvp-card {
        background: linear-gradient(135deg, #ffd700, #ffb300);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #1a237e;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Индикаторы риска */
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 5px solid #f9a825;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Статистика в реальном времени */
    .stat-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 2px solid #1E88E5;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a237e;
    }
    
    .stat-medal {
        font-size: 1.2rem;
        margin-left: 0.5rem;
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
            # Добавляем расширенную статистику
            'statistics': {}
        }
        
        total_chars = len(text)
        if total_chars == 0:
            return results
        
        # Считаем слова для нормировки
        words = text.split()
        total_words = len(words)
        
        # Проходим по тексту (теперь без ограничения)
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
        
        # Добавляем статистику
        results['statistics'] = {
            'mean_per_1000': results['non_latin_per_1000'],
            'median_per_1000': results['non_latin_per_1000'],  # Для одного значения совпадает
            'max_value': results['non_latin_total'],
            'distribution': {
                'greek': results['greek_count'],
                'cyrillic': results['cyrillic_count'],
                'arabic': results['arabic_count']
            }
        }
        
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
            'statistics': {}
        }
        
        if not sentences:
            return results
        
        dash_counts = []
        
        for sent in sentences:
            if not sent or len(sent.strip()) == 0:
                continue
                
            dash_count = sent.count(self.em_dash)
            if dash_count > 0:
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
        
        if sentences:
            results['percentage_heavy'] = (len(results['heavy_sentences']) / len(sentences)) * 100
        
        # Статистика по тире
        if dash_counts:
            results['statistics'] = {
                'mean_per_sentence': float(np.mean(dash_counts)),
                'median_per_sentence': float(np.median(dash_counts)),
                'max_per_sentence': float(np.max(dash_counts)),
                'std_per_sentence': float(np.std(dash_counts)),
                'total_dashes': sum(dash_counts),
                'sentences_with_dashes': len(dash_counts)
            }
        
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
            
            # ДОБАВЛЕНО: новые слова по заданию
            'pathway', 'pathways',  # pathway/signaling
            'signaling', 'signals',  # signaling
            'Collectively',  # с большой буквы в начале предложения
            'collectively',  # также в середине предложения
            'manifest',  # manifest
            'paradigm', 'paradigms',  # paradigm (уже было, но добавим множественное)
            
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
            'statistics': {}
        }
        
        if not text or not sentences:
            return results
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        # Подсчет каждой фразы
        phrase_freq = []
        for phrase in self.ai_phrases:
            count = text_lower.count(phrase.lower())
            if count > 0:
                results['phrase_counts'][phrase] = count
                phrase_freq.append(count)
        
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
        
        # Статистика по фразам
        if phrase_freq:
            results['statistics'] = {
                'mean_per_phrase': float(np.mean(phrase_freq)),
                'median_per_phrase': float(np.median(phrase_freq)),
                'max_per_phrase': float(np.max(phrase_freq)),
                'std_per_phrase': float(np.std(phrase_freq)),
                'total_unique_phrases': len(results['phrase_counts']),
                'total_phrase_occurrences': sum(phrase_freq)
            }
        
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
            'statistics': {}
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
        
        # Расширенная статистика
        results['statistics'] = {
            'mean': results['mean_length'],
            'median': float(np.median(sent_lengths)),
            'max': float(np.max(sent_lengths)),
            'min': float(np.min(sent_lengths)),
            'std': results['std_length'],
            'cv': results['cv'],
            'q25': float(q25),
            'q75': float(q75),
            'iqr': results['iqr'],
            'total_sentences': len(sent_lengths)
        }
        
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
            'statistics': {}
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
        
        # Статистика
        results['statistics'] = {
            'passive_mean_per_sentence': results['passive_percentage'] / 100 if results['total_sentences'] > 0 else 0,
            'nominalizations_per_1000': results['nominalizations_per_1000'],
            'modals_per_1000': results['modals_per_1000'],
            'epistemic_per_1000': results['epistemic_per_1000'],
            'boosters_per_1000': results['boosters_per_1000']
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
            'statistics': {}
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
        
        # Нормировка на 1000 слов
        if results['total_words'] > 0:
            results['hedging_per_1000'] = (results['hedging_count'] * 1000) / results['total_words']
            results['certainty_per_1000'] = (results['certainty_count'] * 1000) / results['total_words']
            results['personal_per_1000'] = (results['personal_count'] * 1000) / results['total_words']
            
            if results['certainty_count'] > 0:
                results['hedging_ratio'] = results['hedging_count'] / results['certainty_count']
        
        # Статистика
        results['statistics'] = {
            'hedging_mean_freq': float(np.mean(hedging_counts)) if hedging_counts else 0,
            'hedging_median_freq': float(np.median(hedging_counts)) if hedging_counts else 0,
            'hedging_max_freq': float(np.max(hedging_counts)) if hedging_counts else 0,
            'certainty_mean_freq': float(np.mean(certainty_counts)) if certainty_counts else 0,
            'personal_mean_freq': float(np.mean(personal_counts)) if personal_counts else 0,
            'hedging_per_1000': results['hedging_per_1000'],
            'certainty_per_1000': results['certainty_per_1000'],
            'hedging_ratio': results['hedging_ratio']
        }
        
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
            'statistics': {}
        }
        
        if not self.available or len(text) < 500:
            return results
        
        paragraphs = self.split_paragraphs(text)
        if len(paragraphs) < 3:
            return results
        
        results['paragraphs'] = paragraphs[:10]  # Ограничиваем
        
        # Статистика по длине абзацев
        para_lengths = [len(p.split()) for p in paragraphs]
        results['statistics'] = {
            'paragraph_count': len(paragraphs),
            'mean_length': float(np.mean(para_lengths)),
            'median_length': float(np.median(para_lengths)),
            'max_length': float(np.max(para_lengths)),
            'min_length': float(np.min(para_lengths)),
            'std_length': float(np.std(para_lengths)),
            'cv_length': float(np.std(para_lengths) / np.mean(para_lengths)) if np.mean(para_lengths) > 0 else 0
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
                results['statistics']['intra_similarity_mean'] = results['intra_paragraph_similarity']
                results['statistics']['intra_similarity_std'] = float(np.std(intra_similarities))
            
            # Межабзацная похожесть
            if len(para_embeddings) > 1:
                inter_similarities = []
                for i in range(len(para_embeddings) - 1):
                    sim = np.dot(para_embeddings[i], para_embeddings[i+1]) / (
                        np.linalg.norm(para_embeddings[i]) * np.linalg.norm(para_embeddings[i+1]))
                    inter_similarities.append(sim)
                results['inter_paragraph_similarity'] = float(np.mean(inter_similarities))
                results['statistics']['inter_similarity_mean'] = results['inter_paragraph_similarity']
                results['statistics']['inter_similarity_std'] = float(np.std(inter_similarities)) if len(inter_similarities) > 1 else 0
            
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
                results['statistics']['transition_mean'] = results['transition_smoothness']
                results['statistics']['transition_std'] = float(np.std(transition_sims)) if len(transition_sims) > 1 else 0
            
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
    """Анализ перплексии (активирован)"""
    
    def __init__(self):
        self.available = TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
        if self.available:
            try:
                # Используем маленькую модель для экономии ресурсов
                self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
                self.model = AutoModelForCausalLM.from_pretrained('gpt2')
            except:
                self.available = False
    
    def analyze(self, text: str) -> Dict:
        """Анализ перплексии текста"""
        results = {
            'perplexities': [],
            'mean_perplexity': 0,
            'median_perplexity': 0,
            'perplexity_variance': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'note': 'Perplexity analysis active',
            'statistics': {}
        }
        
        if not self.available or len(text) < 100:
            results['note'] = 'Perplexity analysis unavailable or text too short'
            results['available'] = False
            return results
        
        try:
            # Разбиваем текст на чанки для анализа
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = ""
            
            for sent in sentences[:50]:  # Ограничиваем для производительности
                if len(current_chunk) + len(sent) < 500:
                    current_chunk += sent + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent + ". "
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Вычисляем перплексию для каждого чанка
            perplexities = []
            for chunk in chunks[:10]:  # Ограничиваем количество чанков
                if len(chunk.strip()) < 50:
                    continue
                    
                inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
            
            if perplexities:
                results['perplexities'] = perplexities
                results['mean_perplexity'] = float(np.mean(perplexities))
                results['median_perplexity'] = float(np.median(perplexities))
                results['perplexity_variance'] = float(np.var(perplexities))
                
                results['statistics'] = {
                    'mean': results['mean_perplexity'],
                    'median': results['median_perplexity'],
                    'std': float(np.std(perplexities)),
                    'min': float(np.min(perplexities)),
                    'max': float(np.max(perplexities)),
                    'q25': float(np.percentile(perplexities, 25)),
                    'q75': float(np.percentile(perplexities, 75))
                }
                
                # Оценка риска на основе перплексии
                # Низкая перплексия (< 30) может указывать на ИИ-текст
                risk_score = 0
                confidence = 0.5
                
                if results['mean_perplexity'] < 30:
                    risk_score = 3
                    confidence = 0.8
                    results['risk_level'] = 'high'
                elif results['mean_perplexity'] < 50:
                    risk_score = 2
                    confidence = 0.6
                    results['risk_level'] = 'medium'
                elif results['mean_perplexity'] < 80:
                    risk_score = 1
                    confidence = 0.5
                    results['risk_level'] = 'low'
                
                results['risk_score'] = risk_score
                results['confidence'] = confidence
            
            results['available'] = True
            
        except Exception as e:
            results['note'] = f'Error in perplexity analysis: {str(e)}'
            results['available'] = False
        
        return results


class SemanticAnalyzer:
    """Анализ семантической близости (активирован)"""
    
    def __init__(self):
        self.available = SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        if self.available:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.available = False
    
    def analyze(self, sentences: List[str]) -> Dict:
        """Семантический анализ предложений"""
        results = {
            'similarities': [],
            'mean_similarity': 0,
            'similarity_variance': 0,
            'risk_level': 'none',
            'risk_score': 0,
            'confidence': 0,
            'note': 'Semantic analysis active',
            'statistics': {},
            'available': self.available
        }
        
        if not self.available or len(sentences) < 5:
            results['note'] = 'Semantic analysis unavailable or too few sentences'
            return results
        
        try:
            # Берем выборку предложений для анализа
            sample_sents = [s for s in sentences if len(s.split()) > 5][:30]
            if len(sample_sents) < 5:
                return results
            
            # Получаем эмбеддинги
            embeddings = self.model.encode(sample_sents)
            
            # Вычисляем попарные сходства
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    similarities.append(sim)
            
            if similarities:
                results['similarities'] = similarities[:100]  # Ограничиваем
                results['mean_similarity'] = float(np.mean(similarities))
                results['similarity_variance'] = float(np.var(similarities))
                
                results['statistics'] = {
                    'mean': results['mean_similarity'],
                    'median': float(np.median(similarities)),
                    'std': float(np.std(similarities)),
                    'min': float(np.min(similarities)),
                    'max': float(np.max(similarities)),
                    'q25': float(np.percentile(similarities, 25)),
                    'q75': float(np.percentile(similarities, 75))
                }
                
                # Оценка риска: высокая семантическая близость (>0.7) может указывать на ИИ
                risk_score = 0
                confidence = 0.5
                
                if results['mean_similarity'] > 0.7:
                    risk_score = 3
                    confidence = 0.8
                    results['risk_level'] = 'high'
                elif results['mean_similarity'] > 0.6:
                    risk_score = 2
                    confidence = 0.6
                    results['risk_level'] = 'medium'
                elif results['mean_similarity'] > 0.5:
                    risk_score = 1
                    confidence = 0.5
                    results['risk_level'] = 'low'
                
                results['risk_score'] = risk_score
                results['confidence'] = confidence
            
        except Exception as e:
            results['note'] = f'Error in semantic analysis: {str(e)}'
            results['available'] = False
        
        return results


class ParenthesisAnalyzer:
    """Анализ длинных пояснений в скобках (НОВЫЙ МОДУЛЬ)"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str) -> Dict:
        """
        Анализирует содержание в круглых скобках
        Длинные пояснения (>=5 слов) - признак человеческого текста
        """
        results = {
            'total_parentheses': 0,
            'short_parentheses': 0,  # <5 слов
            'long_parentheses': 0,    # >=5 слов
            'long_percentage': 0,
            'long_examples': [],
            'average_words_per_parenthesis': 0,
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'statistics': {}
        }
        
        if not text:
            return results
        
        # Находим все содержимое в круглых скобках
        pattern = r'\(([^()]+)\)'  # Простой паттерн для скобок
        matches = re.findall(pattern, text)
        
        if not matches:
            return results
        
        results['total_parentheses'] = len(matches)
        
        word_counts = []
        long_examples = []
        
        for match in matches[:50]:  # Ограничиваем для производительности
            words = match.split()
            word_count = len(words)
            word_counts.append(word_count)
            
            if word_count >= 5:
                results['long_parentheses'] += 1
                if len(long_examples) < 5:
                    long_examples.append({
                        'text': match[:100] + '...' if len(match) > 100 else match,
                        'word_count': word_count
                    })
            else:
                results['short_parentheses'] += 1
        
        results['long_examples'] = long_examples
        
        if results['total_parentheses'] > 0:
            results['long_percentage'] = (results['long_parentheses'] / results['total_parentheses']) * 100
            results['average_words_per_parenthesis'] = float(np.mean(word_counts))
        
        # Статистика
        if word_counts:
            results['statistics'] = {
                'mean_words': float(np.mean(word_counts)),
                'median_words': float(np.median(word_counts)),
                'max_words': float(np.max(word_counts)),
                'min_words': float(np.min(word_counts)),
                'std_words': float(np.std(word_counts)),
                'q25_words': float(np.percentile(word_counts, 25)),
                'q75_words': float(np.percentile(word_counts, 75)),
                'long_percentage': results['long_percentage']
            }
        
        # Оценка риска: ЧЕЛОВЕК использует длинные пояснения в скобках
        # Поэтому высокий процент длинных скобок -> НИЗКИЙ риск ИИ
        risk_score = 0
        confidence = 0.5
        
        if results['long_percentage'] > 30:
            # Много длинных пояснений - признак человека
            risk_score = 0
            confidence = 0.8
            results['risk_level'] = 'low'
        elif results['long_percentage'] > 15:
            # Умеренное количество
            risk_score = 1
            confidence = 0.6
            results['risk_level'] = 'low'
        elif results['long_percentage'] > 5:
            # Мало длинных пояснений
            risk_score = 2
            confidence = 0.7
            results['risk_level'] = 'medium'
        else:
            # Почти нет длинных пояснений - подозрительно (ИИ избегает сложных пояснений)
            risk_score = 3
            confidence = 0.8
            results['risk_level'] = 'high'
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        return results


class PunctuationAnalyzer:
    """Анализ пунктуации ! ? ; (НОВЫЙ МОДУЛЬ)"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str, sentences: List[str]) -> Dict:
        """
        Анализирует использование ! ? ;
        Редкие знаки и сложные конструкции с ; - признак человека
        """
        results = {
            'exclamation_count': 0,
            'exclamation_per_1000': 0,
            'question_count': 0,
            'question_per_1000': 0,
            'semicolon_count': 0,
            'semicolon_per_1000': 0,
            'semicolon_contexts': [],
            'complex_semicolon_structures': 0,  # когда после ; идет короткое предложение
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'statistics': {}
        }
        
        if not text:
            return results
        
        words = text.split()
        total_words = len(words)
        
        # Подсчет знаков
        results['exclamation_count'] = text.count('!')
        results['question_count'] = text.count('?')
        results['semicolon_count'] = text.count(';')
        
        # Нормировка на 1000 слов
        if total_words > 0:
            results['exclamation_per_1000'] = (results['exclamation_count'] * 1000) / total_words
            results['question_per_1000'] = (results['question_count'] * 1000) / total_words
            results['semicolon_per_1000'] = (results['semicolon_count'] * 1000) / total_words
        
        # Анализ контекста для ;
        semicolon_pattern = r'([^;]+);\s*([^;]+)'
        semicolon_matches = re.findall(semicolon_pattern, text)
        
        semicolon_lengths = []
        for before, after in semicolon_matches[:10]:
            before_words = len(before.split())
            after_words = len(after.split())
            semicolon_lengths.append({
                'before': before_words,
                'after': after_words
            })
            
            # Если после ; идет очень короткое предложение (<=5 слов) - это сложная структура
            if after_words <= 5 and before_words > 10:
                results['complex_semicolon_structures'] += 1
                if len(results['semicolon_contexts']) < 5:
                    results['semicolon_contexts'].append({
                        'before': before[:50] + '...' if len(before) > 50 else before,
                        'after': after[:50] + '...' if len(after) > 50 else after,
                        'before_words': before_words,
                        'after_words': after_words
                    })
        
        # Статистика
        results['statistics'] = {
            'exclamation_per_1000': results['exclamation_per_1000'],
            'question_per_1000': results['question_per_1000'],
            'semicolon_per_1000': results['semicolon_per_1000'],
            'complex_semicolon_structures': results['complex_semicolon_structures'],
            'total_punctuation_marks': results['exclamation_count'] + results['question_count'] + results['semicolon_count']
        }
        
        if semicolon_lengths:
            after_lengths = [item['after'] for item in semicolon_lengths]
            results['statistics']['semicolon_after_mean'] = float(np.mean(after_lengths))
            results['statistics']['semicolon_after_median'] = float(np.median(after_lengths))
        
        # Оценка риска: наличие сложных знаков - признак человека
        risk_score = 0
        confidence = 0.5
        
        # Если есть сложные структуры с ; - признак человека
        if results['complex_semicolon_structures'] > 3:
            risk_score = 0  # Низкий риск
            confidence = 0.8
            results['risk_level'] = 'low'
        elif results['complex_semicolon_structures'] > 0:
            risk_score = 1
            confidence = 0.6
            results['risk_level'] = 'low'
        
        # Если есть восклицательные знаки (редко в науке) - признак человека
        if results['exclamation_count'] > 0:
            risk_score = max(0, risk_score - 1)  # Уменьшаем риск
            confidence = min(confidence + 0.1, 1.0)
            if results['risk_level'] == 'none' or results['risk_level'] == 'high':
                results['risk_level'] = 'medium'
        
        # Если есть вопросительные знаки (вопросы в тексте) - признак человека
        if results['question_count'] > 5:
            risk_score = max(0, risk_score - 1)
            confidence = min(confidence + 0.1, 1.0)
        
        # Если вообще нет пунктуации кроме точек - подозрительно
        if (results['exclamation_count'] == 0 and results['question_count'] == 0 and 
            results['semicolon_count'] == 0 and total_words > 1000):
            risk_score = 3
            confidence = 0.7
            results['risk_level'] = 'high'
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        return results


class ApostropheAnalyzer:
    """Анализ апострофов 's (НОВЫЙ МОДУЛЬ)"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str) -> Dict:
        """
        Анализирует использование апострофов (Parkinson's, pathway's)
        Частота использования - признак человеческого текста
        """
        results = {
            'apostrophe_count': 0,
            'apostrophe_per_1000': 0,
            'apostrophe_examples': [],
            'unique_apostrophe_words': set(),
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'statistics': {}
        }
        
        if not text:
            return results
        
        words = text.split()
        total_words = len(words)
        
        # Паттерн для поиска слово + 's
        pattern = r"\b(\w+)'s\b"
        matches = re.findall(pattern, text)
        
        results['apostrophe_count'] = len(matches)
        results['unique_apostrophe_words'] = set(matches)
        
        # Нормировка на 1000 слов
        if total_words > 0:
            results['apostrophe_per_1000'] = (results['apostrophe_count'] * 1000) / total_words
        
        # Примеры
        examples = list(set(matches))[:10]  # Уникальные примеры
        results['apostrophe_examples'] = [f"{word}'s" for word in examples]
        
        # Статистика
        results['statistics'] = {
            'apostrophe_per_1000': results['apostrophe_per_1000'],
            'unique_apostrophe_words': len(results['unique_apostrophe_words']),
            'total_apostrophes': results['apostrophe_count'],
            'avg_apostrophe_per_unique': results['apostrophe_count'] / len(results['unique_apostrophe_words']) if len(results['unique_apostrophe_words']) > 0 else 0
        }
        
        # Оценка риска: наличие апострофов - признак человеческого текста
        risk_score = 3  # Высокий риск по умолчанию
        confidence = 0.5
        
        if results['apostrophe_per_1000'] > 2:
            # Много апострофов - явный признак человека
            risk_score = 0
            confidence = 0.9
            results['risk_level'] = 'low'
        elif results['apostrophe_per_1000'] > 1:
            # Умеренное количество
            risk_score = 1
            confidence = 0.7
            results['risk_level'] = 'low'
        elif results['apostrophe_per_1000'] > 0.5:
            # Мало апострофов
            risk_score = 2
            confidence = 0.6
            results['risk_level'] = 'medium'
        elif results['apostrophe_count'] > 0:
            # Очень мало апострофов
            risk_score = 2
            confidence = 0.5
            results['risk_level'] = 'medium'
        else:
            # Нет апострофов - подозрительно для научной статьи
            risk_score = 3
            confidence = 0.7
            results['risk_level'] = 'high'
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        return results


class EnumerationAnalyzer:
    """Анализ строгих перечислений из трех элементов (НОВЫЙ МОДУЛЬ)"""
    
    def __init__(self):
        pass
    
    def analyze(self, text: str) -> Dict:
        """
        Анализирует строгие перечисления вида: X, Y, and Z
        Такие конструкции характерны для человеческого письма
        """
        results = {
            'enumeration_count': 0,
            'enumeration_per_1000': 0,
            'enumeration_examples': [],
            'enumeration_contexts': [],
            'risk_score': 0,
            'confidence': 0,
            'risk_level': 'none',
            'statistics': {}
        }
        
        if not text:
            return results
        
        words = text.split()
        total_words = len(words)
        
        # Паттерн для поиска: слово, слово, and слово
        # Упрощенный паттерн для демонстрации
        pattern = r'\b(\w+),\s+(\w+),\s+and\s+(\w+)\b'
        matches = re.findall(pattern, text)
        
        # Также ищем с запятыми внутри (для фраз из нескольких слов)
        pattern2 = r'([^,]+),\s+([^,]+),\s+and\s+([^\.]+)'
        matches2 = re.findall(pattern2, text)
        
        results['enumeration_count'] = len(matches) + len(matches2)
        
        # Нормировка на 1000 слов
        if total_words > 0:
            results['enumeration_per_1000'] = (results['enumeration_count'] * 1000) / total_words
        
        # Примеры
        for match in matches[:5]:
            example = f"{match[0]}, {match[1]}, and {match[2]}"
            results['enumeration_examples'].append(example)
            results['enumeration_contexts'].append({
                'items': list(match),
                'type': 'simple'
            })
        
        for match in matches2[:5]:
            example = f"{match[0].strip()}, {match[1].strip()}, and {match[2].strip()}"
            if len(example) < 100:
                results['enumeration_examples'].append(example)
                results['enumeration_contexts'].append({
                    'items': [m.strip() for m in match],
                    'type': 'complex'
                })
        
        # Статистика
        results['statistics'] = {
            'enumeration_per_1000': results['enumeration_per_1000'],
            'total_enumerations': results['enumeration_count'],
            'unique_enumerations': len(results['enumeration_examples'])
        }
        
        # Оценка риска: наличие строгих перечислений - признак человека
        risk_score = 3
        confidence = 0.5
        
        if results['enumeration_per_1000'] > 0.5:
            # Много перечислений - признак человека
            risk_score = 0
            confidence = 0.8
            results['risk_level'] = 'low'
        elif results['enumeration_per_1000'] > 0.2:
            # Умеренное количество
            risk_score = 1
            confidence = 0.6
            results['risk_level'] = 'low'
        elif results['enumeration_count'] > 0:
            # Мало перечислений
            risk_score = 2
            confidence = 0.5
            results['risk_level'] = 'medium'
        else:
            # Нет перечислений - подозрительно
            risk_score = 3
            confidence = 0.7
            results['risk_level'] = 'high'
        
        results['risk_score'] = risk_score
        results['confidence'] = confidence
        
        return results


class IntegratedRiskScorer:
    """Интегральная оценка риска на основе всех модулей"""
    
    def __init__(self):
        # Веса модулей (обновлено с учетом новых модулей)
        self.weights = {
            'unicode': 0.05,        # 5% - уменьшили
            'dashes': 0.05,          # 5% - уменьшили
            'phrases': 0.10,         # 10% - умеренно
            'burstiness': 0.08,      # 8% - умеренно
            'grammar': 0.10,         # 10% - грамматика
            'hedging': 0.15,         # 15% - ключевой
            'paragraph': 0.10,       # 10% - абзацы
            'perplexity': 0.07,      # 7% - перплексия
            'semantic': 0.07,        # 7% - семантика
            'parenthesis': 0.07,     # 7% - скобки (НОВЫЙ)
            'punctuation': 0.06,      # 6% - пунктуация (НОВЫЙ)
            'apostrophe': 0.05,       # 5% - апострофы (НОВЫЙ)
            'enumeration': 0.05       # 5% - перечисления (НОВЫЙ)
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
        
        # Для определения MVP (лучшего/худшего модуля)
        max_contribution = -1
        max_module = None
        min_contribution = 1
        min_module = None
        
        for module, weight in self.weights.items():
            if module in results and results[module]:
                data = results[module]
                if 'risk_score' in data and data['risk_score'] is not None:
                    # Нормируем risk_score (0-3) к 0-1
                    norm_score = data['risk_score'] / 3.0
                    
                    # Учитываем уверенность модуля
                    confidence = data.get('confidence', 0.5)
                    
                    contribution = norm_score * weight * 100
                    
                    module_score = {
                        'module': module,
                        'raw_score': data['risk_score'],
                        'norm_score': norm_score,
                        'weight': weight,
                        'confidence': confidence,
                        'contribution': contribution
                    }
                    
                    module_scores.append(module_score)
                    weighted_score += norm_score * weight
                    total_confidence += confidence * weight
                    
                    # Отслеживаем максимумы и минимумы
                    if contribution > max_contribution:
                        max_contribution = contribution
                        max_module = module
                    
                    if contribution < min_contribution and norm_score > 0:
                        min_contribution = contribution
                        min_module = module
        
        # Итоговая оценка 0-100
        final_score = weighted_score * 100
        
        # Корректировка на уверенность
        if total_confidence > 0:
            final_score = final_score * (0.5 + 0.5 * total_confidence)
        
        # Определяем уровень риска
        risk_level = 'unknown'
        if final_score < 20:
            risk_level = 'low'
        elif final_score < 40:
            risk_level = 'medium-low'
        elif final_score < 60:
            risk_level = 'medium'
        elif final_score < 80:
            risk_level = 'medium-high'
        else:
            risk_level = 'high'
        
        # Определяем медали для модулей
        medal_mapping = {}
        if module_scores:
            sorted_scores = sorted(module_scores, key=lambda x: x['contribution'], reverse=True)
            if len(sorted_scores) > 0:
                medal_mapping[sorted_scores[0]['module']] = 'gold'
            if len(sorted_scores) > 1:
                medal_mapping[sorted_scores[1]['module']] = 'silver'
            if len(sorted_scores) > 2:
                medal_mapping[sorted_scores[2]['module']] = 'bronze'
        
        return {
            'final_score': final_score,
            'risk_level': risk_level,
            'weighted_score': weighted_score,
            'total_confidence': total_confidence,
            'module_scores': module_scores,
            'mvp_module': max_module,
            'mvp_contribution': max_contribution,
            'medal_mapping': medal_mapping
        }


class DocumentProcessor:
    """Обработчик загруженных документов"""
    
    @staticmethod
    def cut_at_references(text: str) -> str:
        """
        Обрезает текст до раздела References/Bibliography
        НОВЫЙ МЕТОД
        """
        if not text:
            return text
        
        # Список маркеров начала списка литературы
        reference_markers = [
            r'\nreferences\s*\n',
            r'\nreferences\s*$',
            r'\nbibliography\s*\n',
            r'\nbibliography\s*$',
            r'\nliterature cited\s*\n',
            r'\nliterature cited\s*$',
            r'\nworks cited\s*\n',
            r'\nworks cited\s*$',
            r'\n cited references\s*\n',
            r'\n cited references\s*$',
            r'\nreference list\s*\n',
            r'\nreference list\s*$',
            r'\nreferences and notes\s*\n',
            r'\nreferences and notes\s*$',
            r'\n☐ references\s*\n',  # Специальные символы
            r'\nREFERENCES\s*\n',
            r'\nBIBLIOGRAPHY\s*\n',
            r'\nLITERATURE CITED\s*\n'
        ]
        
        # Ищем первый маркер
        min_pos = len(text)
        for marker in reference_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                pos = match.start()
                if pos < min_pos:
                    min_pos = pos
        
        if min_pos < len(text):
            return text[:min_pos].strip()
        
        return text.strip()
    
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
                    # Обрезаем до References
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
    st.markdown('<h1 class="main-header">🏆 ЧЕМПИОНАТ AI-ДЕТЕКЦИИ 2026</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span style="background: linear-gradient(135deg, #1E88E5, #ffd700); 
                     padding: 0.5rem 2rem; 
                     border-radius: 50px;
                     color: white;
                     font-weight: bold;
                     text-transform: uppercase;
                     box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
            ⚡ Прямая трансляция анализа ⚡
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Боковая панель с настройками (УБРАНЫ ЧЕКБОКСЫ МОДУЛЕЙ)
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">🏆</span>
            <h3>СЕЗОН 2026</h3>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_depth = st.select_slider(
            "Глубина анализа",
            options=['Быстрый', 'Стандартный', 'Глубокий'],
            value='Стандартный'
        )
        
        st.markdown("---")
        
        # Информация о модулях (все активны)
        st.markdown("""
        ### 📋 СОСТАВ КОМАНДЫ
        
        **Основной состав:**
        - ⚽ Хеджинг (капитан)
        - ⚽ ИИ-фразы
        - ⚽ Burstiness
        - ⚽ Грамматика
        - ⚽ Абзацный анализ
        
        **Новые игроки 2026:**
        - 🆕 Длинные скобки
        - 🆕 Пунктуация (! ? ;)
        - 🆕 Апострофы ('s)
        - 🆕 Перечисления (X, Y, and Z)
        
        **Спецприглашения:**
        - 🔬 Перплексия
        - 🔬 Семантика
        """)
        
        st.markdown("---")
        
        # Турнирная таблица весов (обновленная)
        st.markdown("""
        ### ⚖️ ВЕСА МОДУЛЕЙ 2026
        
        | Модуль | Вес |
        |--------|-----|
        | Хеджинг | 15% |
        | ИИ-фразы | 10% |
        | Грамматика | 10% |
        | Абзацы | 10% |
        | Burstiness | 8% |
        | Перплексия | 7% |
        | Семантика | 7% |
        | Скобки | 7% |
        | Пунктуация | 6% |
        | Тире | 5% |
        | Unicode | 5% |
        | Апострофы | 5% |
        | Перечисления | 5% |
        """)
        
        st.markdown("---")
        
        st.info(
            "🏆 Все модули в активном составе!\n\n"
            "Анализируется полный текст до раздела References."
        )
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "📄 Загрузите файл статьи (поддерживаются .docx и .doc)", 
        type=['docx', 'doc'],
        help="Анализируется весь текст до раздела References"
    )
    
    if uploaded_file is not None:
        # Показываем информацию о файле
        file_details = {
            "Имя файла": uploaded_file.name,
            "Тип": uploaded_file.type or "application/octet-stream",
            "Размер": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <span class="stat-label">ЗАГРУЖЕН ФАЙЛ</span>
                <span class="stat-value">{uploaded_file.name}</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("▶️ НАЧАТЬ МАТЧ"):
                st.rerun()
        
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
            
            # Шаг 2: Предобработка
            status_text.text("Предобработка текста...")
            progress_bar.progress(10)
            text = DocumentProcessor.preprocess(text)
            
            # Шаг 3: Сегментация на предложения (простая)
            status_text.text("Сегментация текста...")
            progress_bar.progress(15)
            sentences = DocumentProcessor.split_sentences_simple(text)
            
            st.success(f"✅ Текст загружен: {len(text)} символов, {len(sentences)} предложений (обрезано до References)")
            
            # Создаем вкладки для результатов
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🏆 ТАБЛО", 
                "⚽ СОСТАВ", 
                "📊 СТАТИСТИКА",
                "🎯 ИНТЕГРАЛ",
                "🆕 НОВИЧКИ 2026",
                "📝 ТЕКСТ"
            ])
            
            # Хранилище результатов
            results = {}
            risk_scores = []
            
            # =================================================================
            # ВСЕ МОДУЛИ АКТИВИРОВАНЫ (УБРАНЫ ЧЕКБОКСЫ)
            # =================================================================
            
            # Модуль 1: Unicode-артефакты
            with st.spinner("Анализ Unicode-артефактов..."):
                detector = UnicodeArtifactDetector()
                results['unicode'] = detector.analyze(text)
                risk_scores.append(results['unicode']['risk_score'])
            progress_bar.progress(10)
            
            # Модуль 2: Множественные тире
            with st.spinner("Анализ множественных тире..."):
                detector = DashAnalyzer()
                results['dashes'] = detector.analyze(sentences)
                risk_scores.append(results['dashes']['risk_score'])
            progress_bar.progress(15)
            
            # Модуль 3: ИИ-фразы
            with st.spinner("Поиск ИИ-фраз (обновленный список 2025-2026)..."):
                detector = AIPhraseDetector()
                results['phrases'] = detector.analyze(text, sentences)
                risk_scores.append(results['phrases']['risk_score'])
            progress_bar.progress(20)
            
            # Модуль 4: Burstiness
            with st.spinner("Анализ вариативности предложений..."):
                detector = BurstinessAnalyzer()
                results['burstiness'] = detector.analyze(sentences)
                if 'error' not in results['burstiness']:
                    risk_scores.append(results['burstiness']['risk_score'])
            progress_bar.progress(25)
            
            # Модуль 5: Грамматика
            with st.spinner("Грамматический анализ..."):
                detector = GrammarAnalyzer()
                results['grammar'] = detector.analyze(text)
                risk_scores.append(results['grammar']['risk_score'])
            progress_bar.progress(30)
            
            # Модуль 6: Хеджинг
            with st.spinner("Анализ хеджинга (капитан команды)..."):
                detector = HedgingAnalyzer()
                results['hedging'] = detector.analyze(text)
                risk_scores.append(results['hedging']['risk_score'])
            progress_bar.progress(35)
            
            # Модуль 7: Абзацный анализ
            with st.spinner("Анализ на уровне абзацев..."):
                detector = ParagraphAnalyzer()
                results['paragraph'] = detector.analyze(text, sentences)
                if 'error' not in results['paragraph']:
                    risk_scores.append(results['paragraph']['risk_score'])
            progress_bar.progress(40)
            
            # Модуль 8: Perplexity (АКТИВИРОВАН)
            with st.spinner("Анализ перплексии..."):
                detector = PerplexityAnalyzer()
                results['perplexity'] = detector.analyze(text)
                if 'error' not in results['perplexity'] and results['perplexity'].get('available', False):
                    risk_scores.append(results['perplexity']['risk_score'])
            progress_bar.progress(45)
            
            # Модуль 9: Семантика (АКТИВИРОВАН)
            with st.spinner("Семантический анализ..."):
                detector = SemanticAnalyzer()
                results['semantic'] = detector.analyze(sentences)
                if 'error' not in results['semantic'] and results['semantic'].get('available', False):
                    risk_scores.append(results['semantic']['risk_score'])
            progress_bar.progress(50)
            
            # Модуль 10: Parenthesis (НОВЫЙ)
            with st.spinner("Анализ длинных пояснений в скобках..."):
                detector = ParenthesisAnalyzer()
                results['parenthesis'] = detector.analyze(text)
                risk_scores.append(results['parenthesis']['risk_score'])
            progress_bar.progress(55)
            
            # Модуль 11: Punctuation (НОВЫЙ)
            with st.spinner("Анализ пунктуации ! ? ;..."):
                detector = PunctuationAnalyzer()
                results['punctuation'] = detector.analyze(text, sentences)
                risk_scores.append(results['punctuation']['risk_score'])
            progress_bar.progress(60)
            
            # Модуль 12: Apostrophe (НОВЫЙ)
            with st.spinner("Анализ апострофов 's..."):
                detector = ApostropheAnalyzer()
                results['apostrophe'] = detector.analyze(text)
                risk_scores.append(results['apostrophe']['risk_score'])
            progress_bar.progress(65)
            
            # Модуль 13: Enumeration (НОВЫЙ)
            with st.spinner("Анализ перечислений X, Y, and Z..."):
                detector = EnumerationAnalyzer()
                results['enumeration'] = detector.analyze(text)
                risk_scores.append(results['enumeration']['risk_score'])
            progress_bar.progress(70)
            
            # Интегральная оценка
            scorer = IntegratedRiskScorer()
            integrated_results = scorer.calculate(results)
            
            # Завершение
            progress_bar.progress(100)
            status_text.text("✅ МАТЧ ЗАВЕРШЕН!")
            
            # =================================================================
            # Вкладка 1: ТАБЛО (Спортивный дизайн)
            # =================================================================
            with tab1:
                # Табло как на стадионе
                final_score = integrated_results['final_score']
                risk_level = integrated_results['risk_level']
                
                # Определяем цвет и текст для табло
                if risk_level == 'low':
                    bg_color = 'linear-gradient(135deg, #1a237e, #0d47a1)'
                    medal_emoji = '🥇'
                    risk_text = 'НИЗКИЙ РИСК'
                elif risk_level == 'medium-low':
                    bg_color = 'linear-gradient(135deg, #1a237e, #2e7d32)'
                    medal_emoji = '🥈'
                    risk_text = 'УМЕРЕННО-НИЗКИЙ'
                elif risk_level == 'medium':
                    bg_color = 'linear-gradient(135deg, #1a237e, #f9a825)'
                    medal_emoji = '🥉'
                    risk_text = 'СРЕДНИЙ РИСК'
                elif risk_level == 'medium-high':
                    bg_color = 'linear-gradient(135deg, #1a237e, #f57c00)'
                    medal_emoji = '🏅'
                    risk_text = 'УМЕРЕННО-ВЫСОКИЙ'
                else:
                    bg_color = 'linear-gradient(135deg, #1a237e, #c62828)'
                    medal_emoji = '⚠️'
                    risk_text = 'ВЫСОКИЙ РИСК'
                
                st.markdown(f"""
                <div class="scoreboard" style="background: {bg_color};">
                    <div class="scoreboard-title">🏆 ФИНАЛЬНЫЙ СЧЕТ 🏆</div>
                    <div class="scoreboard-number">{final_score:.1f}</div>
                    <div class="scoreboard-label">{medal_emoji} {risk_text} {medal_emoji}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Карточки игроков (модулей) с наивысшим вкладом
                st.markdown("### ⚡ MVP МАТЧА")
                
                if integrated_results['mvp_module']:
                    mvp = integrated_results['mvp_module']
                    mvp_score = integrated_results['mvp_contribution']
                    
                    # Человеческое название модуля
                    module_names = {
                        'hedging': 'ХЕДЖИНГ', 'phrases': 'ИИ-ФРАЗЫ', 'burstiness': 'BURSTINESS',
                        'grammar': 'ГРАММАТИКА', 'paragraph': 'АБЗАЦЫ', 'unicode': 'UNICODE',
                        'dashes': 'ТИРЕ', 'perplexity': 'ПЕРПЛЕКСИЯ', 'semantic': 'СЕМАНТИКА',
                        'parenthesis': 'СКОБКИ', 'punctuation': 'ПУНКТУАЦИЯ', 
                        'apostrophe': 'АПОСТРОФЫ', 'enumeration': 'ПЕРЕЧИСЛЕНИЯ'
                    }
                    
                    st.markdown(f"""
                    <div class="mvp-card">
                        <span style="font-size: 2rem;">🏆</span><br>
                        <span style="font-size: 1.5rem;">{module_names.get(mvp, mvp.upper())}</span><br>
                        <span style="font-size: 3rem; font-weight: bold;">{mvp_score:.1f}%</span><br>
                        <span>вклад в итоговый результат</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Статистика матча
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="stat-box">
                        <span class="stat-label">СЫГРАНО МОДУЛЕЙ</span>
                        <span class="stat-value">{}</span>
                    </div>
                    """.format(len(risk_scores)), unsafe_allow_html=True)
                
                with col2:
                    avg_risk = float(np.mean(risk_scores)) if risk_scores else 0
                    st.markdown("""
                    <div class="stat-box">
                        <span class="stat-label">СРЕДНИЙ РИСК</span>
                        <span class="stat-value">{:.2f}</span>
                    </div>
                    """.format(avg_risk), unsafe_allow_html=True)
                
                with col3:
                    max_risk = float(np.max(risk_scores)) if risk_scores else 0
                    st.markdown("""
                    <div class="stat-box">
                        <span class="stat-label">МАКСИМАЛЬНЫЙ</span>
                        <span class="stat-value">{:.2f}</span>
                    </div>
                    """.format(max_risk), unsafe_allow_html=True)
                
                with col4:
                    confidence = integrated_results['total_confidence'] * 100
                    st.markdown("""
                    <div class="stat-box">
                        <span class="stat-label">УВЕРЕННОСТЬ</span>
                        <span class="stat-value">{:.0f}%</span>
                    </div>
                    """.format(confidence), unsafe_allow_html=True)
            
            # =================================================================
            # Вкладка 2: СОСТАВ (карточки игроков)
            # =================================================================
            with tab2:
                st.header("⚽ СОСТАВ КОМАНДЫ 2026")
                
                # Создаем карточки для каждого модуля
                module_names = {
                    'hedging': ('ХЕДЖИНГ', 'Капитан команды', '🛡️'),
                    'phrases': ('ИИ-ФРАЗЫ', 'Нападающий', '🤖'),
                    'burstiness': ('BURSTINESS', 'Полузащитник', '📊'),
                    'grammar': ('ГРАММАТИКА', 'Защитник', '📚'),
                    'paragraph': ('АБЗАЦЫ', 'Защитник', '📑'),
                    'unicode': ('UNICODE', 'Резерв', '🔣'),
                    'dashes': ('ТИРЕ', 'Резерв', '➖'),
                    'perplexity': ('ПЕРПЛЕКСИЯ', 'Легионер', '🧮'),
                    'semantic': ('СЕМАНТИКА', 'Легионер', '🔄'),
                    'parenthesis': ('СКОБКИ', 'Новичок 2026', '📌'),
                    'punctuation': ('ПУНКТУАЦИЯ', 'Новичок 2026', '❗'),
                    'apostrophe': ('АПОСТРОФЫ', 'Новичок 2026', '🔤'),
                    'enumeration': ('ПЕРЕЧИСЛЕНИЯ', 'Новичок 2026', '🔢')
                }
                
                # Создаем сетку карточек
                cols = st.columns(3)
                col_idx = 0
                
                for module_key, (display_name, position, emoji) in module_names.items():
                    if module_key in results:
                        data = results[module_key]
                        risk_score = data.get('risk_score', 0)
                        
                        # Определяем медаль
                        medal = ''
                        if integrated_results['medal_mapping'].get(module_key) == 'gold':
                            medal = '🥇'
                        elif integrated_results['medal_mapping'].get(module_key) == 'silver':
                            medal = '🥈'
                        elif integrated_results['medal_mapping'].get(module_key) == 'bronze':
                            medal = '🥉'
                        
                        # Определяем цвет риска
                        if risk_score >= 3:
                            risk_color = '#c62828'
                            risk_text = 'ВЫСОКИЙ'
                        elif risk_score >= 2:
                            risk_color = '#f9a825'
                            risk_text = 'СРЕДНИЙ'
                        elif risk_score >= 1:
                            risk_color = '#2e7d32'
                            risk_text = 'НИЗКИЙ'
                        else:
                            risk_color = '#1E88E5'
                            risk_text = 'НЕТ'
                        
                        with cols[col_idx % 3]:
                            st.markdown(f"""
                            <div class="player-card">
                                <div class="player-name">{medal} {emoji} {display_name} {medal}</div>
                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{position}</div>
                                <div class="player-score" style="color: {risk_color};">{risk_score}/3</div>
                                <div class="player-medal">{risk_text}</div>
                                <div style="font-size: 0.8rem; margin-top: 0.5rem;">Уверенность: {data.get('confidence', 0.5):.0%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        col_idx += 1
            
            # =================================================================
            # Вкладка 3: СТАТИСТИКА (подробная статистика)
            # =================================================================
            with tab3:
                st.header("📊 ДЕТАЛЬНАЯ СТАТИСТИКА")
                
                # Сводная таблица
                summary_data = []
                for key, data in results.items():
                    if isinstance(data, dict):
                        risk_level = data.get('risk_level', 'unknown')
                        risk_score = data.get('risk_score', 0)
                        confidence = data.get('confidence', 0.5)
                        
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
                        elif 'apostrophe_per_1000' in data:
                            main_value = f"{data['apostrophe_per_1000']:.2f}/1000"
                        elif 'enumeration_per_1000' in data:
                            main_value = f"{data['enumeration_per_1000']:.2f}/1000"
                        elif 'long_percentage' in data:
                            main_value = f"{data['long_percentage']:.1f}%"
                        
                        # Человеческое название
                        display_name = module_names.get(key, (key.capitalize(), '', ''))[0]
                        
                        summary_data.append({
                            'Модуль': display_name,
                            'Уровень риска': risk_level.upper(),
                            'Оценка': f"{risk_score}/3",
                            'Уверенность': f"{confidence:.0%}",
                            'Значение': main_value
                        })
                
                if summary_data:
                    df = pd.DataFrame(summary_data)
                    
                    # Цветовая кодировка
                    def color_risk(val):
                        if 'HIGH' in val:
                            return 'background-color: #ffebee; color: #c62828; font-weight: bold;'
                        elif 'MEDIUM' in val or 'MEDIUM-HIGH' in val or 'MEDIUM-LOW' in val:
                            return 'background-color: #fff8e1; color: #f9a825; font-weight: bold;'
                        elif 'LOW' in val:
                            return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold;'
                        return ''
                    
                    styled_df = df.style.applymap(color_risk, subset=['Уровень риска'])
                    st.dataframe(styled_df, use_container_width=True)
                
                # Расширенная статистика по каждому модулю
                st.subheader("📈 РАСШИРЕННАЯ СТАТИСТИКА")
                
                for key, data in results.items():
                    if 'statistics' in data and data['statistics']:
                        display_name = module_names.get(key, (key.capitalize(), '', ''))[0]
                        with st.expander(f"{display_name} - детализация"):
                            stats = data['statistics']
                            # Создаем DataFrame для статистики
                            stats_df = pd.DataFrame([stats])
                            st.dataframe(stats_df.T.rename(columns={0: 'Значение'}), use_container_width=True)
            
            # =================================================================
            # Вкладка 4: ИНТЕГРАЛЬНАЯ ОЦЕНКА
            # =================================================================
            with tab4:
                st.header("🎯 ИНТЕГРАЛЬНАЯ ОЦЕНКА")
                
                # Турнирная таблица
                st.markdown("""
                <div class="tournament-table">
                    <h3 style="text-align: center;">📊 ТУРНИРНАЯ ТАБЛИЦА МОДУЛЕЙ</h3>
                """, unsafe_allow_html=True)
                
                if integrated_results['module_scores']:
                    # Сортируем по вкладу
                    sorted_modules = sorted(integrated_results['module_scores'], 
                                          key=lambda x: x['contribution'], reverse=True)
                    
                    module_names_rev = {v: k for k, (v, _, _) in module_names.items()}
                    
                    for i, module in enumerate(sorted_modules):
                        mod_key = module['module']
                        display_name = module_names.get(mod_key, (mod_key.capitalize(), '', ''))[0]
                        contribution = module['contribution']
                        
                        # Определяем прогресс бар
                        bar_width = min(contribution * 2, 100)  # Для наглядности масштабируем
                        
                        # Определяем медаль
                        medal = ''
                        if i == 0:
                            medal = '🥇'
                        elif i == 1:
                            medal = '🥈'
                        elif i == 2:
                            medal = '🥉'
                        
                        st.markdown(f"""
                        <div class="tournament-row">
                            <span class="tournament-rank">{i+1}.</span>
                            <span class="tournament-player">{medal} {display_name}</span>
                            <div class="tournament-bar">
                                <div class="tournament-progress" style="width: {bar_width}%;"></div>
                            </div>
                            <span class="tournament-value">{contribution:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Диаграмма вклада
                st.subheader("📊 ВКЛАД МОДУЛЕЙ В ИТОГОВЫЙ РЕЗУЛЬТАТ")
                
                if integrated_results['module_scores']:
                    module_df = pd.DataFrame(integrated_results['module_scores'])
                    
                    # Добавляем человеческие названия
                    module_df['display_name'] = module_df['module'].apply(
                        lambda x: module_names.get(x, (x.capitalize(), '', ''))[0]
                    )
                    
                    # Сортируем по вкладу
                    module_df = module_df.sort_values('contribution', ascending=True)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=module_df['contribution'],
                            y=module_df['display_name'],
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
                        title="Вклад модулей в итоговый риск (в %)",
                        xaxis_title="Вклад в итоговую оценку (%)",
                        yaxis_title="Модуль",
                        height=500,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # =================================================================
            # Вкладка 5: НОВИЧКИ 2026 (детальный анализ новых модулей)
            # =================================================================
            with tab5:
                st.header("🆕 НОВЫЕ ИГРОКИ СЕЗОНА 2026")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Parenthesis анализ
                    if 'parenthesis' in results:
                        st.subheader("📌 Длинные пояснения в скобках")
                        p = results['parenthesis']
                        
                        st.metric("Всего скобок", p['total_parentheses'])
                        st.metric("Длинных (≥5 слов)", p['long_parentheses'])
                        st.metric("Процент длинных", f"{p['long_percentage']:.1f}%")
                        st.metric("Средняя длина", f"{p.get('average_words_per_parenthesis', 0):.1f} слов")
                        
                        if p['long_examples']:
                            st.write("**Примеры длинных пояснений:**")
                            for ex in p['long_examples'][:3]:
                                st.info(f"📝 {ex['text']} ({ex['word_count']} слов)")
                        
                        st.metric("Оценка риска", f"{p['risk_score']}/3")
                        st.metric("Уверенность", f"{p['confidence']:.0%}")
                    
                    # Apostrophe анализ
                    if 'apostrophe' in results:
                        st.subheader("🔤 Апострофы 's")
                        a = results['apostrophe']
                        
                        st.metric("Всего апострофов", a['apostrophe_count'])
                        st.metric("На 1000 слов", f"{a['apostrophe_per_1000']:.2f}")
                        st.metric("Уникальных слов", len(a['unique_apostrophe_words']))
                        
                        if a['apostrophe_examples']:
                            st.write("**Примеры:**")
                            st.write(", ".join(a['apostrophe_examples'][:8]))
                        
                        st.metric("Оценка риска", f"{a['risk_score']}/3")
                        st.metric("Уверенность", f"{a['confidence']:.0%}")
                
                with col_b:
                    # Punctuation анализ
                    if 'punctuation' in results:
                        st.subheader("❗ Пунктуация ! ? ;")
                        pu = results['punctuation']
                        
                        st.metric("Восклицательных знаков", f"{pu['exclamation_per_1000']:.2f}/1000")
                        st.metric("Вопросительных знаков", f"{pu['question_per_1000']:.2f}/1000")
                        st.metric("Точка с запятой", f"{pu['semicolon_per_1000']:.2f}/1000")
                        st.metric("Сложных структур с ;", pu['complex_semicolon_structures'])
                        
                        if pu['semicolon_contexts']:
                            st.write("**Примеры сложных ;:**")
                            for ctx in pu['semicolon_contexts'][:2]:
                                st.info(f"📋 {ctx['before']} ; {ctx['after']}")
                        
                        st.metric("Оценка риска", f"{pu['risk_score']}/3")
                        st.metric("Уверенность", f"{pu['confidence']:.0%}")
                    
                    # Enumeration анализ
                    if 'enumeration' in results:
                        st.subheader("🔢 Перечисления X, Y, and Z")
                        e = results['enumeration']
                        
                        st.metric("Всего перечислений", e['enumeration_count'])
                        st.metric("На 1000 слов", f"{e['enumeration_per_1000']:.2f}")
                        
                        if e['enumeration_examples']:
                            st.write("**Примеры:**")
                            for ex in e['enumeration_examples'][:3]:
                                st.info(f"📋 {ex}")
                        
                        st.metric("Оценка риска", f"{e['risk_score']}/3")
                        st.metric("Уверенность", f"{e['confidence']:.0%}")
            
            # =================================================================
            # Вкладка 6: ТЕКСТ
            # =================================================================
            with tab6:
                st.header("📝 ИСХОДНЫЙ ТЕКСТ")
                
                # Показываем подозрительные предложения
                if 'dashes' in results and results['dashes']['heavy_sentences']:
                    with st.expander("⚠️ Предложения с множественными тире", expanded=False):
                        for i, ex in enumerate(results['dashes']['examples'][:5]):
                            st.markdown(f"**{i+1}.** {ex}")
                
                if 'phrases' in results and results['phrases']['repeated_phrases']:
                    with st.expander("⚠️ Контекст повторяющихся фраз", expanded=False):
                        for phrase_data in results['phrases']['repeated_phrases'][:3]:
                            phrase = phrase_data['phrase']
                            # Ищем контекст
                            pattern = re.compile(r'([^.]*?' + re.escape(phrase) + r'[^.]*\.)', re.IGNORECASE)
                            matches = pattern.findall(text[:5000])
                            for match in matches[:2]:
                                st.markdown(f"- *{phrase}*: ...{match}...")
                
                # Показываем текст
                st.subheader("Текст статьи (обрезан до References)")
                
                # Разбиваем на страницы для удобства
                text_pages = [text[i:i+5000] for i in range(0, len(text), 5000)]
                page_num = st.selectbox("Страница", range(1, len(text_pages)+1))
                
                text_sample = text_pages[page_num-1]
                
                # Подсвечиваем подозрительные символы
                if 'unicode' in results and results['unicode']['suspicious_chunks']:
                    for chunk in results['unicode']['suspicious_chunks'][:3]:
                        char = chunk['char']
                        # Простая подсветка в тексте
                        if char in text_sample:
                            text_sample = text_sample.replace(char, f"**`{char}`**")
                
                st.markdown(text_sample)
                
                st.info(f"Страница {page_num} из {len(text_pages)}. Всего символов: {len(text)}")
        
        except Exception as e:
            st.error(f"Ошибка при обработке: {str(e)}")
            st.exception(e)
    
    else:
        # Информация о приложении
        st.info("👆 Загрузите файл для начала матча!")
        
        # Превью состава
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="player-card">
                <div class="player-name">⚽ ОСНОВА</div>
                <div>Хеджинг (капитан)</div>
                <div>ИИ-фразы</div>
                <div>Burstiness</div>
                <div>Грамматика</div>
                <div>Абзацы</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="player-card">
                <div class="player-name">🆕 НОВИЧКИ 2026</div>
                <div>Длинные скобки</div>
                <div>Пунктуация ! ? ;</div>
                <div>Апострофы 's</div>
                <div>Перечисления</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="player-card">
                <div class="player-name">🔬 ЛЕГИОНЕРЫ</div>
                <div>Перплексия</div>
                <div>Семантика</div>
                <div>Unicode</div>
                <div>Тире</div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ О сезоне 2026"):
            st.markdown("""
            ### 🏆 ЧЕМПИОНАТ AI-ДЕТЕКЦИИ 2026
            
            **Новые игроки в составе:**
            - **Длинные скобки** - анализ пояснений в скобках (>=5 слов - признак человека)
            - **Пунктуация** - использование ! ? ; (сложные структуры - признак человека)
            - **Апострофы** - частота 's (Parkinson's, pathway's - признак человека)
            - **Перечисления** - строгие структуры X, Y, and Z (признак человека)
            
            **Обновленная статистика:**
            - Для всех метрик вычисляются mean, median, max, std
            - Полный анализ текста до References
            - Все модули всегда активны
            
            **Система оценки:**
            - Табло с итоговым счетом 0-100
            - Медали для лучших модулей
            - Турнирная таблица с прогресс-барами
            - MVP матча
            """)


if __name__ == "__main__":
    main()
