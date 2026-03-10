"""
Streamlit приложение для анализа научных статей на предмет использования ИИ
Исправленная версия с учетом совместимости версий и оптимизацией для Streamlit Cloud
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
        
        # Все подозрительные символы
        self.all_suspicious = self.sup_sub_chars.union(self.fullwidth_digits)
    
    def analyze(self, text: str) -> Dict:
        """Анализирует текст на наличие Unicode-артефактов"""
        results = {
            'sup_sub_count': 0,
            'fullwidth_count': 0,
            'homoglyph_count': 0,
            'density_per_10k': 0,
            'suspicious_chunks': [],
            'risk_level': 'none',
            'risk_score': 0
        }
        
        total_chars = len(text)
        if total_chars == 0:
            return results
        
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
        
        # Плотность на 10,000 символов
        total_suspicious = results['sup_sub_count'] + results['fullwidth_count']
        results['density_per_10k'] = (total_suspicious * 10000) / max_chars if max_chars > 0 else 0
        
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
            'risk_score': 0
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
        
        # Ограничиваем количество для отображения
        results['sentences_with_multiple_dashes'] = results['sentences_with_multiple_dashes'][:10]
        results['heavy_sentences'] = results['heavy_sentences'][:5]
        
        return results


class AIPhraseDetector:
    """Детектор характерных ИИ-фраз и штампов (уровень 1)"""
    
    def __init__(self):
        # Основные ИИ-фразы
        self.ai_phrases = [
            # "Модные" слова
            'delve into', 'testament to', 'pivotal role', 'sheds light',
            'in the tapestry', 'in the realm', 'underscores', 'harnesses',
            'ever-evolving landscape', 'nuanced understanding', 'robust framework',
            'holistic approach', 'paradigm shift', 'cutting-edge',
            
            # Устойчивые связки
            'it is worth noting', 'it is important to note',
            'it should be noted', 'as we delve deeper',
            'in the context of', 'with respect to', 'in terms of',
            
            # Избыточные переходы
            'moreover', 'furthermore', 'in addition', 'consequently',
            'therefore', 'thus', 'hence', 'nonetheless', 'nevertheless',
        ]
        
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
        
        if not text or not sentences:
            return results
        
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
        
        if len(sentences) > 0:
            results['transition_density'] = (results['transition_count'] * 1000) / len(sentences)
        
        # Оценка риска
        risk_score = 0
        
        if len(results['repeated_phrases']) > 2:
            risk_score += 2
        elif len(results['repeated_phrases']) > 0:
            risk_score += 1
        
        if results['transition_density'] > self.transition_threshold:
            risk_score += 2
        elif results['transition_density'] > self.transition_threshold * 0.7:
            risk_score += 1
        
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
        results = {
            'sentence_lengths': [],
            'mean_length': 0,
            'std_length': 0,
            'cv': 0,
            'iqr': 0,
            'burstiness': 'unknown',
            'risk_level': 'none',
            'risk_score': 0
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


class GrammarAnalyzer:
    """Анализ грамматических особенностей (без spacy, упрощенная версия)"""
    
    def analyze(self, text: str) -> Dict:
        """Упрощенный анализ грамматики без spacy"""
        results = {
            'passive_indicators': 0,
            'nominalization_count': 0,
            'total_sentences': 0,
            'passive_percentage': 0,
            'nominalizations_per_1000': 0,
            'examples': [],
            'risk_level': 'none',
            'risk_score': 0
        }
        
        if not text:
            return results
        
        # Простая сегментация на предложения
        sentences = re.split(r'[.!?]+', text)
        results['total_sentences'] = len([s for s in sentences if len(s.strip()) > 10])
        
        # Поиск индикаторов пассива (was/were + ed/en)
        passive_pattern = r'\b(was|were|is|are|been|being)\s+(\w+ed|\w+en)\b'
        passive_matches = re.findall(passive_pattern, text.lower())
        results['passive_indicators'] = len(passive_matches)
        
        # Суффиксы номинализации
        nominalization_suffixes = ['tion', 'ment', 'ance', 'ence', 'ing', 'ity', 'ism']
        words = text.lower().split()
        
        for word in words:
            for suffix in nominalization_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    results['nominalization_count'] += 1
                    break
        
        if results['total_sentences'] > 0:
            results['passive_percentage'] = (results['passive_indicators'] / results['total_sentences']) * 100
        
        total_words = len(words)
        if total_words > 0:
            results['nominalizations_per_1000'] = (results['nominalization_count'] * 1000) / total_words
        
        # Оценка риска
        risk_score = 0
        if results['passive_percentage'] > 40:
            risk_score += 2
        elif results['passive_percentage'] > 30:
            risk_score += 1
        
        if results['nominalizations_per_1000'] > 20:
            risk_score += 2
        elif results['nominalizations_per_1000'] > 12:
            risk_score += 1
        
        if risk_score >= 3:
            results['risk_level'] = 'high'
        elif risk_score >= 2:
            results['risk_level'] = 'medium'
        elif risk_score >= 1:
            results['risk_level'] = 'low'
        
        results['risk_score'] = risk_score
        
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
            'to some extent', 'in some cases', 'relatively'
        ]
        
        self.personal_pronouns = ['we', 'our', 'us', 'i', 'my']
        
    def analyze(self, text: str) -> Dict:
        """Анализирует использование хеджинга"""
        results = {
            'hedging_count': 0,
            'hedging_per_1000': 0,
            'personal_count': 0,
            'personal_per_1000': 0,
            'total_words': 0,
            'risk_level': 'none',
            'risk_score': 0
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
        
        # Подсчет личных местоимений
        for pronoun in self.personal_pronouns:
            results['personal_count'] += sum(1 for w in words if w == pronoun)
        
        # Нормировка на 1000 слов
        if results['total_words'] > 0:
            results['hedging_per_1000'] = (results['hedging_count'] * 1000) / results['total_words']
            results['personal_per_1000'] = (results['personal_count'] * 1000) / results['total_words']
        
        # Оценка риска
        risk_score = 0
        if results['hedging_per_1000'] < 3:
            risk_score += 3
        elif results['hedging_per_1000'] < 5:
            risk_score += 2
        elif results['hedging_per_1000'] < 7:
            risk_score += 1
        
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
            'risk_level': 'none',
            'risk_score': 0,
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
            'risk_level': 'none',
            'risk_score': 0,
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
                    results['mean_similarity'] = 0.8  # Высокая близость
                    results['risk_level'] = 'high'
                    results['risk_score'] = 3
                elif cv < 0.5:
                    results['mean_similarity'] = 0.6
                    results['risk_level'] = 'medium'
                    results['risk_score'] = 2
                else:
                    results['mean_similarity'] = 0.3
                    results['risk_level'] = 'low'
                    results['risk_score'] = 1
        
        return results


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
            options=['Быстрый', 'Стандартный'],
            value='Стандартный'
        )
        
        st.markdown("---")
        st.markdown("**Активные модули:**")
        
        modules = {
            'unicode': st.checkbox("Unicode-артефакты", value=True),
            'dashes': st.checkbox("Множественные тире", value=True),
            'phrases': st.checkbox("ИИ-фразы", value=True),
            'burstiness': st.checkbox("Burstiness", value=True),
            'grammar': st.checkbox("Грамматика", value=True),
            'hedging': st.checkbox("Хеджинг", value=True),
            'perplexity': st.checkbox("Perplexity (если доступно)", value=False),
            'semantic': st.checkbox("Семантика (если доступно)", value=False)
        }
        
        st.markdown("---")
        st.info(
            "Примечание: Полный семантический анализ требует дополнительных "
            "библиотек и может быть недоступен в облачной версии."
        )
    
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
            progress_bar.progress(10)
            
            if uploaded_file.name.endswith('.docx'):
                text = DocumentProcessor.read_docx(uploaded_file)
            else:  # .doc
                text = DocumentProcessor.read_doc(uploaded_file)
            
            if not text or len(text.strip()) < 100:
                st.warning("Файл слишком мал или не содержит текста")
                return
            
            # Шаг 2: Предобработка
            status_text.text("Предобработка текста...")
            progress_bar.progress(20)
            text = DocumentProcessor.preprocess(text)
            
            # Шаг 3: Сегментация на предложения (простая)
            status_text.text("Сегментация текста...")
            progress_bar.progress(30)
            sentences = DocumentProcessor.split_sentences_simple(text)
            
            st.success(f"✅ Текст загружен: {len(text)} символов, {len(sentences)} предложений")
            
            # Создаем вкладки для результатов
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Общий отчет", 
                "🔍 Артефакты", 
                "📈 Статистика",
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
                progress_bar.progress(40)
            
            # =================================================================
            # Модуль 2: Множественные тире
            # =================================================================
            if modules['dashes']:
                with st.spinner("Анализ множественных тире..."):
                    detector = DashAnalyzer()
                    results['dashes'] = detector.analyze(sentences)
                    risk_scores.append(results['dashes']['risk_score'])
                progress_bar.progress(50)
            
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
                progress_bar.progress(70)
            
            # =================================================================
            # Модуль 5: Грамматика
            # =================================================================
            if modules['grammar']:
                with st.spinner("Грамматический анализ..."):
                    detector = GrammarAnalyzer()
                    results['grammar'] = detector.analyze(text)
                    risk_scores.append(results['grammar']['risk_score'])
                progress_bar.progress(75)
            
            # =================================================================
            # Модуль 6: Хеджинг
            # =================================================================
            if modules['hedging']:
                with st.spinner("Анализ хеджинга..."):
                    detector = HedgingAnalyzer()
                    results['hedging'] = detector.analyze(text)
                    risk_scores.append(results['hedging']['risk_score'])
                progress_bar.progress(80)
            
            # =================================================================
            # Модуль 7: Perplexity
            # =================================================================
            if modules['perplexity']:
                with st.spinner("Анализ перплексии..."):
                    detector = PerplexityAnalyzer()
                    results['perplexity'] = detector.analyze(text)
                    if 'error' not in results['perplexity']:
                        risk_scores.append(results['perplexity']['risk_score'])
                progress_bar.progress(85)
            
            # =================================================================
            # Модуль 8: Семантика
            # =================================================================
            if modules['semantic']:
                with st.spinner("Семантический анализ..."):
                    detector = SemanticAnalyzer()
                    results['semantic'] = detector.analyze(sentences)
                    if 'error' not in results['semantic']:
                        risk_scores.append(results['semantic']['risk_score'])
                progress_bar.progress(90)
            
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
                    
                    # Сводная таблица
                    st.subheader("📋 Сводка по метрикам")
                    summary_data = []
                    for key, data in results.items():
                        if isinstance(data, dict):
                            risk_level = data.get('risk_level', 'unknown')
                            risk_score = data.get('risk_score', 0)
                            
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
                            
                            summary_data.append({
                                'Метрика': key.capitalize(),
                                'Уровень риска': risk_level.upper(),
                                'Оценка': f"{risk_score}/3",
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
                        st.subheader("Unicode-артефакты")
                        u = results['unicode']
                        
                        metrics_cols = st.columns(2)
                        with metrics_cols[0]:
                            st.metric("Плотность на 10k", f"{u['density_per_10k']:.2f}")
                            st.metric("Надстрочные/подстрочные", u['sup_sub_count'])
                        with metrics_cols[1]:
                            st.metric("Fullwidth цифры", u['fullwidth_count'])
                            st.metric("Гомоглифы", u['homoglyph_count'])
                        
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
                    st.subheader("🤖 ИИ-фразы и штампы")
                    p = results['phrases']
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Плотность переходов", f"{p['transition_density']:.2f}/1000 предл.")
                    
                    with col4:
                        st.metric("Повторяющихся фраз", len(p['repeated_phrases']))
                    
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
                        st.subheader("Burstiness")
                        b = results['burstiness']
                        
                        st.metric("Ср. длина предложения", f"{b['mean_length']:.1f} слов")
                        st.metric("Станд. отклонение", f"{b['std_length']:.1f}")
                        st.metric("Коэф. вариации", f"{b['cv']:.3f}")
                        
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
                        st.subheader("📚 Грамматика")
                        g = results['grammar']
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Пассивных конструкций", f"{g['passive_percentage']:.1f}%")
                        with col_b:
                            st.metric("Номинализаций на 1000", f"{g['nominalizations_per_1000']:.1f}")
                        
                        if g['examples']:
                            st.write("**Пример пассива:**")
                            for ex in g['examples'][:2]:
                                st.info(ex)
                
                with stat_cols[1]:
                    if 'hedging' in results:
                        st.subheader("🤔 Хеджинг")
                        h = results['hedging']
                        
                        st.metric("Хеджинг на 1000 слов", f"{h['hedging_per_1000']:.2f}")
                        st.metric("Личные местоимения на 1000", f"{h['personal_per_1000']:.2f}")
                        
                        # Индикатор
                        if h['hedging_per_1000'] < 3:
                            st.warning("Очень низкий уровень хеджинга - текст может быть слишком категоричным")
                        elif h['hedging_per_1000'] < 5:
                            st.info("Умеренный уровень хеджинга")
                        else:
                            st.success("Хороший уровень хеджинга")
                    
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
            # Вкладка 4: Текст
            # =================================================================
            with tab4:
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
        
        with st.expander("ℹ️ О метриках анализа"):
            st.markdown("""
            ### Уровни анализа:
            
            **Уровень 1: Артефакты**
            - **Unicode-гомоглифы**: надстрочные/подстрочные символы, fullwidth цифры, смешение алфавитов
            - **Множественные тире**: ≥3 длинных тире в предложении или ≥2 в предложениях >90 слов
            - **ИИ-фразы**: характерные для LLM выражения и штампы
            
            **Уровень 2: Статистика**
            - **Burstiness**: вариативность длины предложений (низкая = вероятно ИИ)
            - **Пассив и номинализация**: грамматические особенности научного стиля
            - **Хеджинг**: использование слов неуверенности (may, might, suggests)
            
            **Как интерпретировать результаты:**
            - 🟢 **Низкий риск** (0-1): текст похож на человеческий
            - 🟡 **Средний риск** (1-2): есть отдельные признаки ИИ
            - 🔴 **Высокий риск** (2-3): множественные признаки генерации ИИ
            """)


if __name__ == "__main__":
    main()
