"""
Streamlit приложение для анализа научных статей на предмет использования ИИ
Исправленная версия с учетом совместимости версий и оптимизацией для Streamlit Cloud
Версия 2.0 - Добавлены: вероятностная оценка, устойчивые маркеры, анализ абзацев
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

# Проверяем и устанавливаем совместимую версию pydantic если нужно
try:
    import pydantic
    from pydantic import BaseModel
except ImportError:
    pass

# NLP библиотеки с обработкой ошибок импорта
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError as e:
    SPACY_AVAILABLE = False
    st.error(f"Ошибка загрузки spaCy: {e}. Некоторые функции будут недоступны.")

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
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
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
            'confidence': 0
        }
        
        total_chars = len(text)
        if total_chars == 0:
            return results
        
        # Считаем слова для нормировки
        words = text.split()
        total_words = len(words)
        
        # Проходим по тексту (ограничиваем для производительности)
        max_chars = min(total_chars, 50000)
        text_sample = text[:max_chars]
        
        for i, char in enumerate(text_sample):
            # Проверка надстрочных/подстрочных
            if char in self.sup_sub_chars:
                results['sup_sub_count'] += 1
                context = text_sample[max(0, i-20):min(len(text_sample), i+20)]
                results['suspicious_chunks'].append({
                    'char': char,
                    'context': context,
                    'type': 'superscript/subscript'
                })
            
            # Проверка fullwidth цифр
            elif char in self.fullwidth_digits:
                results['fullwidth_count'] += 1
                context = text_sample[max(0, i-20):min(len(text_sample), i+20)]
                results['suspicious_chunks'].append({
                    'char': char,
                    'context': context,
                    'type': 'fullwidth digit'
                })
            
            # Проверка гомоглифов
            elif char in self.homoglyphs and i > 0 and i < len(text_sample)-1:
                surrounding = text_sample[max(0, i-3):i] + text_sample[i+1:min(len(text_sample), i+4)]
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
        results['density_per_10k'] = (total_suspicious * 10000) / max_chars if max_chars > 0 else 0
        
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
            'confidence': 0
        }
        
        if not sentences:
            return results
        
        for sent in sentences:
            if not sent or len(sent.strip()) == 0:
                continue
                
            dash_count = sent.count(self.em_dash)
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
            'risk_level': 'none'
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
            'confidence': 0
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
            'confidence': 0
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
            'confidence': 0
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
            'note': 'Paragraph analysis requires sentence-transformers'
        }
        
        if not self.available or len(text) < 500:
            return results
        
        paragraphs = self.split_paragraphs(text)
        if len(paragraphs) < 3:
            return results
        
        results['paragraphs'] = paragraphs[:10]  # Ограничиваем
        
        # Если нет модели, используем простые метрики
        if not self.model:
            # Простая метрика: вариативность длины абзацев
            para_lengths = [len(p.split()) for p in paragraphs]
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
            'note': 'Perplexity analysis requires transformers library'
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
            'note': 'Semantic analysis requires sentence-transformers'
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
        # Веса модулей (сумма = 100)
        self.weights = {
            'unicode': 0.08,      # 8% - почти исчез
            'dashes': 0.10,        # 10% - встречается
            'phrases': 0.15,       # 15% - упали, но не до нуля
            'burstiness': 0.12,    # 12% - остается полезным
            'grammar': 0.15,       # 15% - грамматика важна
            'hedging': 0.20,       # 20% - стойкий маркер
            'paragraph': 0.20,     # 20% - абзацный анализ
            'perplexity': 0.0,     # 0% - отключено
            'semantic': 0.0        # 0% - отключено
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
        elif final_score < 50:
            risk_level = 'medium-low'
        elif final_score < 75:
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
            'unicode': st.checkbox("Unicode-артефакты + нелатинские символы", value=True),
            'dashes': st.checkbox("Множественные тире", value=True),
            'phrases': st.checkbox("ИИ-фразы (обновлено 2025-2026)", value=True),
            'burstiness': st.checkbox("Burstiness", value=True),
            'grammar': st.checkbox("Грамматика + модальность", value=True),
            'hedging': st.checkbox("Хеджинг (ключевой маркер)", value=True),
            'paragraph': st.checkbox("Абзацный анализ (если доступно)", value=True),
            'perplexity': st.checkbox("Perplexity (если доступно)", value=False),
            'semantic': st.checkbox("Семантика (если доступно)", value=False)
        }
        
        st.markdown("---")
        st.info(
            "Примечание: Полный семантический и абзацный анализ требует дополнительных "
            "библиотек и может быть недоступен в облачной версии."
        )
        
        st.markdown("---")
        st.markdown("**Интегральная оценка**")
        st.markdown("Используются веса модулей 2025-2026:")
        st.markdown("- Хеджинг: 20%")
        st.markdown("- Абзацный анализ: 20%")
        st.markdown("- Грамматика: 15%")
        st.markdown("- ИИ-фразы: 15%")
        st.markdown("- Burstiness: 12%")
        st.markdown("- Тире: 10%")
        st.markdown("- Unicode: 8%")
    
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
            "Тип": uploaded_file.type or "application/octet-stream",
            "Размер": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("📄 Информация о файле:", file_details)
        with col2:
            if st.button("🔄 Начать анализ"):
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
            
            st.success(f"✅ Текст загружен: {len(text)} символов, {len(sentences)} предложений")
            
            # Создаем вкладки для результатов
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Общий отчет", 
                "🔍 Артефакты", 
                "📈 Статистика",
                "📊 Интегральная оценка",
                "📝 Текст"
            ])
            
            # Хранилище результатов
            results = {}
            risk_scores = []
            
            # =================================================================
            # Модуль 1: Unicode-артефакты
            # =================================================================
            if modules['unicode']:
                with st.spinner("Анализ Unicode-артефактов и нелатинских символов..."):
                    detector = UnicodeArtifactDetector()
                    results['unicode'] = detector.analyze(text)
                    risk_scores.append(results['unicode']['risk_score'])
                progress_bar.progress(25)
            
            # =================================================================
            # Модуль 2: Множественные тире
            # =================================================================
            if modules['dashes']:
                with st.spinner("Анализ множественных тире..."):
                    detector = DashAnalyzer()
                    results['dashes'] = detector.analyze(sentences)
                    risk_scores.append(results['dashes']['risk_score'])
                progress_bar.progress(35)
            
            # =================================================================
            # Модуль 3: ИИ-фразы
            # =================================================================
            if modules['phrases']:
                with st.spinner("Поиск ИИ-фраз (обновленный список 2025-2026)..."):
                    detector = AIPhraseDetector()
                    results['phrases'] = detector.analyze(text, sentences)
                    risk_scores.append(results['phrases']['risk_score'])
                progress_bar.progress(45)
            
            # =================================================================
            # Модуль 4: Burstiness
            # =================================================================
            if modules['burstiness']:
                with st.spinner("Анализ вариативности предложений..."):
                    detector = BurstinessAnalyzer()
                    results['burstiness'] = detector.analyze(sentences)
                    if 'error' not in results['burstiness']:
                        risk_scores.append(results['burstiness']['risk_score'])
                progress_bar.progress(55)
            
            # =================================================================
            # Модуль 5: Грамматика
            # =================================================================
            if modules['grammar']:
                with st.spinner("Грамматический анализ (модальность, номинализации)..."):
                    detector = GrammarAnalyzer()
                    results['grammar'] = detector.analyze(text)
                    risk_scores.append(results['grammar']['risk_score'])
                progress_bar.progress(65)
            
            # =================================================================
            # Модуль 6: Хеджинг
            # =================================================================
            if modules['hedging']:
                with st.spinner("Анализ хеджинга (ключевой маркер)..."):
                    detector = HedgingAnalyzer()
                    results['hedging'] = detector.analyze(text)
                    risk_scores.append(results['hedging']['risk_score'])
                progress_bar.progress(75)
            
            # =================================================================
            # Модуль 7: Абзацный анализ
            # =================================================================
            if modules['paragraph']:
                with st.spinner("Анализ на уровне абзацев..."):
                    detector = ParagraphAnalyzer()
                    results['paragraph'] = detector.analyze(text, sentences)
                    if 'error' not in results['paragraph']:
                        risk_scores.append(results['paragraph']['risk_score'])
                progress_bar.progress(85)
            
            # =================================================================
            # Модуль 8: Perplexity
            # =================================================================
            if modules['perplexity']:
                with st.spinner("Анализ перплексии..."):
                    detector = PerplexityAnalyzer()
                    results['perplexity'] = detector.analyze(text)
                    if 'error' not in results['perplexity']:
                        risk_scores.append(results['perplexity']['risk_score'])
                progress_bar.progress(90)
            
            # =================================================================
            # Модуль 9: Семантика
            # =================================================================
            if modules['semantic']:
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
            # Вкладка 1: Общий отчет
            # =================================================================
            with tab1:
                st.header("📊 Общая оценка")
                
                # Интегральный риск
                if risk_scores:
                    avg_risk = float(np.mean(risk_scores))
                    max_risk = float(np.max(risk_scores))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
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
                        st.metric("Средний риск (0-3)", f"{avg_risk:.2f}")
                    
                    with col2:
                        st.metric("Максимальный риск", f"{max_risk:.2f}")
                    
                    with col3:
                        st.metric("Активных модулей", len(risk_scores))
                    
                    with col4:
                        st.metric("Интегральная оценка", f"{integrated_results['final_score']:.1f}/100")
                    
                    # Сводная таблица
                    st.subheader("📋 Сводка по метрикам")
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
                            
                            summary_data.append({
                                'Метрика': key.capitalize(),
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
                                return 'background-color: #ffcccc'
                            elif 'MEDIUM' in val:
                                return 'background-color: #fff3cd'
                            elif 'LOW' in val:
                                return 'background-color: #d4edda'
                            return ''
                        
                        styled_df = df.style.applymap(color_risk, subset=['Уровень риска'])
                        st.dataframe(styled_df, use_container_width=True)
                
                else:
                    st.warning("Нет данных для анализа")
            
            # =================================================================
            # Вкладка 2: Артефакты
            # =================================================================
            with tab2:
                st.header("🔍 Артефакты форматирования")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'unicode' in results:
                        st.subheader("Unicode-артефакты и нелатинские символы")
                        u = results['unicode']
                        
                        metrics_cols = st.columns(2)
                        with metrics_cols[0]:
                            st.metric("Плотность на 10k", f"{u['density_per_10k']:.2f}")
                            st.metric("Надстрочные/подстрочные", u['sup_sub_count'])
                            st.metric("Греческие символы", u['greek_count'])
                        with metrics_cols[1]:
                            st.metric("Fullwidth цифры", u['fullwidth_count'])
                            st.metric("Гомоглифы", u['homoglyph_count'])
                            st.metric("Кириллица", u['cyrillic_count'])
                        
                        st.metric("Нелатинские символы на 1000 слов", f"{u['non_latin_per_1000']:.2f}")
                        
                        if u['suspicious_chunks']:
                            st.write("**Примеры подозрительных символов:**")
                            for chunk in u['suspicious_chunks'][:3]:
                                st.code(f"Символ '{chunk['char']}' в контексте: ...{chunk['context']}...")
                
                with col2:
                    if 'dashes' in results:
                        st.subheader("Множественные тире")
                        d = results['dashes']
                        
                        st.metric("Предложений с ≥2 тире", len(d['sentences_with_multiple_dashes']))
                        st.metric("Тяжелые предложения", len(d['heavy_sentences']))
                        st.metric("Процент тяжелых", f"{d['percentage_heavy']:.2f}%")
                        
                        if d['examples']:
                            st.write("**Примеры:**")
                            for i, ex in enumerate(d['examples'][:2]):
                                st.info(f"Пример {i+1}: {ex}")
                
                if 'phrases' in results:
                    st.subheader("🤖 ИИ-фразы и штампы (обновлено 2025-2026)")
                    p = results['phrases']
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Плотность переходов", f"{p['transition_density']:.2f}/1000 слов")
                        st.metric("Метадискурс на 1000 слов", f"{p.get('metadiscourse_per_1000', 0):.2f}")
                    
                    with col4:
                        st.metric("Повторяющихся фраз", len(p['repeated_phrases']))
                        st.metric("Уверенность", f"{p.get('confidence', 0.5):.0%}")
                    
                    if p['top_phrases']:
                        st.write("**Наиболее частые фразы:**")
                        for phrase, count in p['top_phrases'][:5]:
                            st.write(f"- '{phrase}': {count} раз(а)")
                    
                    if p['repeated_phrases']:
                        st.write("**⚠️ Часто повторяющиеся фразы:**")
                        for item in p['repeated_phrases']:
                            st.write(f"- '{item['phrase']}' ({item['count']} раз)")
            
            # =================================================================
            # Вкладка 3: Статистика
            # =================================================================
            with tab3:
                st.header("📈 Статистический анализ")
                
                stat_cols = st.columns(2)
                
                with stat_cols[0]:
                    if 'burstiness' in results and 'error' not in results['burstiness']:
                        st.subheader("Burstiness (вариативность предложений)")
                        b = results['burstiness']
                        
                        st.metric("Ср. длина предложения", f"{b['mean_length']:.1f} слов")
                        st.metric("Станд. отклонение", f"{b['std_length']:.1f}")
                        st.metric("Коэф. вариации", f"{b['cv']:.3f}")
                        st.metric("Уверенность", f"{b.get('confidence', 0.5):.0%}")
                        
                        # Простая гистограмма с plotly
                        if b['sentence_lengths']:
                            fig = go.Figure(data=[go.Histogram(x=b['sentence_lengths'], nbinsx=20)])
                            fig.update_layout(
                                title="Распределение длины предложений",
                                xaxis_title="Длина (слова)",
                                yaxis_title="Частота",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if 'grammar' in results:
                        st.subheader("📚 Грамматика и модальность")
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
                
                with stat_cols[1]:
                    if 'hedging' in results:
                        st.subheader("🤔 Хеджинг (ключевой маркер)")
                        h = results['hedging']
                        
                        col_c, col_d = st.columns(2)
                        with col_c:
                            st.metric("Хеджинг на 1000 слов", f"{h['hedging_per_1000']:.2f}")
                            st.metric("Категоричность на 1000", f"{h.get('certainty_per_1000', 0):.2f}")
                        with col_d:
                            st.metric("Личные местоимения на 1000", f"{h['personal_per_1000']:.2f}")
                            st.metric("Уверенность", f"{h.get('confidence', 0.5):.0%}")
                        
                        # Индикатор
                        if h['hedging_per_1000'] < 3:
                            st.warning("⚠️ Очень низкий уровень хеджинга - сильный сигнал ИИ")
                        elif h['hedging_per_1000'] < 5:
                            st.info("Умеренный уровень хеджинга")
                        else:
                            st.success("✅ Хороший уровень хеджинга (человеческий стиль)")
                    
                    if 'paragraph' in results and 'error' not in results['paragraph']:
                        st.subheader("📑 Абзацный анализ")
                        p = results['paragraph']
                        
                        if 'intra_paragraph_similarity' in p:
                            st.metric("Внутриабзацная похожесть", f"{p['intra_paragraph_similarity']:.3f}")
                            st.metric("Межабзацная похожесть", f"{p.get('inter_paragraph_similarity', 0):.3f}")
                            st.metric("Плавность переходов", f"{p.get('transition_smoothness', 0):.3f}")
                            st.metric("Уверенность", f"{p.get('confidence', 0.5):.0%}")
                            
                            if p.get('intra_paragraph_similarity', 0) > 0.72:
                                st.warning("⚠️ Очень высокая внутриабзацная похожесть - типично для ИИ")
                        
                        st.info(p.get('note', ''))
                    
                    if 'perplexity' in results:
                        st.subheader("📊 Perplexity")
                        pp = results['perplexity']
                        if 'note' in pp:
                            st.info(pp['note'])
                        else:
                            st.metric("Средняя перплексия", f"{pp.get('mean_perplexity', 0):.2f}")
                    
                    if 'semantic' in results:
                        st.subheader("🔄 Семантическая близость")
                        sm = results['semantic']
                        if 'note' in sm:
                            st.info(sm['note'])
                        else:
                            st.metric("Средняя близость", f"{sm.get('mean_similarity', 0):.3f}")
            
            # =================================================================
            # Вкладка 4: Интегральная оценка
            # =================================================================
            with tab4:
                st.header("📊 Интегральная оценка риска")
                
                # Отображаем итоговый результат
                final_score = integrated_results['final_score']
                risk_level = integrated_results['risk_level']
                
                if risk_level == 'low':
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3>🟢 Низкий риск использования ИИ ({final_score:.1f}/100)</h3>
                        <p>Текст демонстрирует характеристики, типичные для человеческого письма:
                        хорошая вариативность, умеренный хеджинг, естественные переходы.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == 'medium-low':
                    st.markdown(f"""
                    <div class="risk-medium">
                        <h3>🟡 Умеренно-низкий риск ({final_score:.1f}/100)</h3>
                        <p>Текст имеет некоторые признаки ИИ-генерации, но в целом похож на человеческий.
                        Рекомендуется дополнительная проверка подозрительных секций.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == 'medium-high':
                    st.markdown(f"""
                    <div class="risk-medium">
                        <h3>🟠 Умеренно-высокий риск ({final_score:.1f}/100)</h3>
                        <p>Текст демонстрирует множественные признаки ИИ-генерации:
                        низкий хеджинг, высокая внутриабзацная похожесть, повторяющиеся фразы.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3>🔴 Высокий риск использования ИИ ({final_score:.1f}/100)</h3>
                        <p>Текст с высокой вероятностью сгенерирован ИИ. Обнаружены сильные маркеры:
                        очень низкий хеджинг, высокая однородность, характерные ИИ-фразы.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Диаграмма вклада модулей
                st.subheader("Вклад модулей в итоговую оценку")
                
                if integrated_results['module_scores']:
                    module_df = pd.DataFrame(integrated_results['module_scores'])
                    
                    # Сортируем по вкладу
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
                        title="Вклад модулей в итоговый риск (в %)",
                        xaxis_title="Вклад в итоговую оценку (%)",
                        yaxis_title="Модуль",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Таблица с деталями
                    st.subheader("Детализация по модулям")
                    module_df['raw_score_display'] = module_df['raw_score'].apply(lambda x: f"{x}/3")
                    module_df['confidence_display'] = module_df['confidence'].apply(lambda x: f"{x:.0%}")
                    module_df['weight_display'] = module_df['weight'].apply(lambda x: f"{x:.1%}")
                    
                    display_df = module_df[['module', 'raw_score_display', 'confidence_display', 
                                           'weight_display', 'contribution']]
                    display_df.columns = ['Модуль', 'Оценка', 'Уверенность', 'Вес', 'Вклад (%)']
                    
                    st.dataframe(display_df, use_container_width=True)
            
            # =================================================================
            # Вкладка 5: Текст
            # =================================================================
            with tab5:
                st.header("📝 Исходный текст")
                
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
                st.subheader("Текст статьи (первые 5000 символов)")
                text_sample = text[:5000] + ("..." if len(text) > 5000 else "")
                
                # Подсвечиваем подозрительные символы
                if 'unicode' in results and results['unicode']['suspicious_chunks']:
                    for chunk in results['unicode']['suspicious_chunks'][:3]:
                        char = chunk['char']
                        # Простая подсветка в тексте
                        if char in text_sample:
                            text_sample = text_sample.replace(char, f"**`{char}`**")
                
                st.markdown(text_sample)
                
                if len(text) > 5000:
                    st.info(f"Показано 5000 из {len(text)} символов. Полный текст доступен в скачанном файле.")
        
        except Exception as e:
            st.error(f"Ошибка при обработке: {str(e)}")
            st.exception(e)
    
    else:
        # Информация о приложении
        st.info("👆 Загрузите файл для начала анализа")
        
        with st.expander("ℹ️ О метриках анализа (обновлено 2025-2026)"):
            st.markdown("""
            ### Уровни анализа:
            
            **Уровень 1: Артефакты**
            - **Unicode-артефакты**: надстрочные/подстрочные символы, fullwidth цифры, гомоглифы
            - **Нелатинские символы**: греческие, кириллица, арабские (частота на 1000 слов)
            - **Множественные тире**: ≥3 длинных тире в предложении
            - **ИИ-фразы 2025-2026**: обновленный список маркеров (landscape, delve, leverage и др.)
            
            **Уровень 2: Статистика**
            - **Burstiness**: вариативность длины предложений (низкая = вероятно ИИ)
            - **Грамматика**: пассив, номинализации, модальные глаголы
            - **Хеджинг**: ключевой маркер - использование слов неуверенности (may, might, suggests)
            
            **Уровень 3: Абзацный анализ (новое)**
            - **Внутриабзацная похожесть**: >0.72 - сильный сигнал ИИ
            - **Плавность переходов**: >0.65 между абзацами - типично для ИИ
            - **Структурная повторяемость**: однородность секций
            
            **Интегральная оценка:**
            - Учитываются веса модулей 2025-2026
            - Итоговая оценка 0-100 с учетом уверенности
            - Детализация вклада каждого модуля
            """)


if __name__ == "__main__":
    main()
