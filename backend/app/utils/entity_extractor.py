"""
Entity Extractor - Extracts structured entities from text
Uses regex patterns and heuristics for fast, local extraction
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import math


@dataclass
class ExtractedEntities:
    """Container for extracted entities from text."""
    # Named entities
    persons: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    
    # Temporal entities
    dates: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    
    # Numeric entities
    monetary_amounts: list[dict] = field(default_factory=list)  # {"value": "1000", "currency": "USD"}
    percentages: list[str] = field(default_factory=list)
    quantities: list[str] = field(default_factory=list)
    
    # Document-specific
    emails: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)
    
    # Keywords
    keywords: list[str] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "persons": self.persons[:10],  # Limit for storage
            "organizations": self.organizations[:10],
            "locations": self.locations[:10],
            "dates": self.dates[:10],
            "monetary_amounts": self.monetary_amounts[:10],
            "percentages": self.percentages[:5],
            "emails": self.emails[:5],
            "urls": self.urls[:5],
            "keywords": self.keywords[:15],
            "key_phrases": self.key_phrases[:10],
        }
    
    def to_searchable_string(self) -> str:
        """Create a searchable string for hybrid search."""
        parts = []
        parts.extend(self.persons)
        parts.extend(self.organizations)
        parts.extend(self.keywords)
        parts.extend(self.key_phrases)
        return " ".join(parts)


class EntityExtractor:
    """
    Fast, local entity extraction using regex and heuristics.
    No external dependencies required.
    """
    
    # Regex patterns
    PATTERNS = {
        # Dates (various formats)
        "date": re.compile(
            r'\b(?:'
            r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|'  # 01/02/2024, 1-2-24
            r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}|'     # 2024-01-02
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|'  # Jan 15, 2024
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|'     # 15 January 2024
            r'(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}'  # Turkish months
            r')\b',
            re.IGNORECASE
        ),
        
        # Monetary amounts
        "money_usd": re.compile(r'\$\s?[\d,]+(?:\.\d{2})?(?:\s?(?:million|billion|M|B|K))?', re.IGNORECASE),
        "money_eur": re.compile(r'€\s?[\d,]+(?:\.\d{2})?(?:\s?(?:million|billion|M|B|K))?', re.IGNORECASE),
        "money_try": re.compile(r'(?:₺|TL|TRY)\s?[\d.,]+(?:\s?(?:milyon|milyar))?', re.IGNORECASE),
        "money_gbp": re.compile(r'£\s?[\d,]+(?:\.\d{2})?(?:\s?(?:million|billion|M|B|K))?', re.IGNORECASE),
        "money_generic": re.compile(r'\b[\d,]+(?:\.\d{2})?\s?(?:USD|EUR|GBP|TRY|dollars?|euros?|pounds?)\b', re.IGNORECASE),
        
        # Percentages
        "percentage": re.compile(r'\b\d+(?:\.\d+)?%|\b(?:percent|yüzde)\s+\d+', re.IGNORECASE),
        
        # Emails
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        
        # URLs
        "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'),
        
        # Phone numbers (international + Turkish format)
        "phone": re.compile(r'(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'),
        
        # Time
        "time": re.compile(r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9]\s?(?:AM|PM|am|pm)|(?:2[0-3]|[01]?[0-9]):[0-5][0-9]\b'),
    }
    
    # Common organization suffixes
    ORG_SUFFIXES = [
        'Inc', 'Inc.', 'LLC', 'Ltd', 'Ltd.', 'Corp', 'Corp.', 'Corporation',
        'Company', 'Co.', 'Co', 'Group', 'Holdings', 'Partners', 'LLP',
        'A.Ş.', 'A.S.', 'AŞ', 'AS', 'Şti.', 'Sti.', 'Limited', 'GmbH', 'AG',
        'Bank', 'Bankası', 'University', 'Üniversitesi', 'Foundation', 'Vakfı'
    ]
    
    # Common title prefixes for persons
    PERSON_PREFIXES = [
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sir', 'Madam',
        'Bay', 'Bayan', 'Sayın', 'Doç.', 'Yrd.', 'Av.'
    ]
    
    # Stop words for keyword extraction
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
        'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
        've', 'ile', 'bir', 'bu', 'şu', 'o', 'de', 'da', 'ki', 'ne', 'için',
        'gibi', 'kadar', 'daha', 'en', 'çok', 'az', 'var', 'yok', 'olan'
    }
    
    def __init__(self):
        self.org_pattern = self._build_org_pattern()
        self.person_pattern = self._build_person_pattern()
    
    def _build_org_pattern(self) -> re.Pattern:
        """Build regex pattern for organizations."""
        suffixes = '|'.join(re.escape(s) for s in self.ORG_SUFFIXES)
        return re.compile(
            rf'\b[A-ZÇĞİÖŞÜ][A-Za-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][A-Za-zçğıöşü]+)*\s+(?:{suffixes})\b'
        )
    
    def _build_person_pattern(self) -> re.Pattern:
        """Build regex pattern for person names with titles."""
        prefixes = '|'.join(re.escape(p) for p in self.PERSON_PREFIXES)
        return re.compile(
            rf'(?:{prefixes})\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+'
        )
    
    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ExtractedEntities object with all found entities
        """
        entities = ExtractedEntities()
        
        if not text or len(text) < 10:
            return entities
        
        # Extract using patterns
        entities.dates = self._extract_pattern("date", text)
        entities.times = self._extract_pattern("time", text)
        entities.percentages = self._extract_pattern("percentage", text)
        entities.emails = self._extract_pattern("email", text)
        entities.urls = self._extract_pattern("url", text)
        entities.phone_numbers = self._extract_pattern("phone", text)
        
        # Extract monetary amounts
        entities.monetary_amounts = self._extract_money(text)
        
        # Extract named entities (organizations, persons)
        entities.organizations = self._extract_organizations(text)
        entities.persons = self._extract_persons(text)
        
        # Extract keywords using TF-IDF-like scoring
        entities.keywords = self._extract_keywords(text)
        entities.key_phrases = self._extract_key_phrases(text)
        
        return entities
    
    def _extract_pattern(self, pattern_name: str, text: str) -> list[str]:
        """Extract matches for a named pattern."""
        pattern = self.PATTERNS.get(pattern_name)
        if not pattern:
            return []
        
        matches = pattern.findall(text)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for m in matches:
            if m.lower() not in seen:
                seen.add(m.lower())
                unique.append(m)
        return unique
    
    def _extract_money(self, text: str) -> list[dict]:
        """Extract monetary amounts with currency."""
        amounts = []
        
        for pattern_name in ["money_usd", "money_eur", "money_try", "money_gbp", "money_generic"]:
            pattern = self.PATTERNS[pattern_name]
            for match in pattern.findall(text):
                currency = "USD"
                if "€" in match or "eur" in match.lower():
                    currency = "EUR"
                elif "₺" in match or "TL" in match or "TRY" in match:
                    currency = "TRY"
                elif "£" in match or "pound" in match.lower():
                    currency = "GBP"
                
                amounts.append({
                    "raw": match.strip(),
                    "currency": currency
                })
        
        # Deduplicate
        seen = set()
        unique = []
        for a in amounts:
            key = a["raw"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        
        return unique
    
    def _extract_organizations(self, text: str) -> list[str]:
        """Extract organization names."""
        matches = self.org_pattern.findall(text)
        
        # Also look for ALL CAPS company names
        caps_pattern = re.compile(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b')
        caps_matches = caps_pattern.findall(text)
        
        # Filter caps matches (avoid common abbreviations)
        common_abbrevs = {'THE', 'AND', 'FOR', 'WITH', 'NOT', 'BUT', 'ARE', 'WAS', 'HAS', 'PDF', 'CEO', 'CTO', 'CFO'}
        caps_matches = [m for m in caps_matches if m not in common_abbrevs and len(m) > 2]
        
        all_orgs = list(set(matches + caps_matches))
        return all_orgs[:10]
    
    def _extract_persons(self, text: str) -> list[str]:
        """Extract person names."""
        # Names with titles
        titled_names = self.person_pattern.findall(text)
        
        # Capitalized name patterns (First Last format)
        name_pattern = re.compile(r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b')
        potential_names = name_pattern.findall(text)
        
        # Filter out common non-names
        non_names = {
            'New York', 'Los Angeles', 'San Francisco', 'United States',
            'North America', 'South America', 'January February', 'March April'
        }
        filtered = [n for n in potential_names if n not in non_names]
        
        all_names = list(set(titled_names + filtered))
        return all_names[:10]
    
    def _extract_keywords(self, text: str, top_n: int = 15) -> list[str]:
        """
        Extract keywords using TF-IDF-like scoring.
        """
        # Tokenize
        words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}\b', text.lower())
        
        # Filter stop words and short words
        words = [w for w in words if w not in self.STOP_WORDS and len(w) > 3]
        
        if not words:
            return []
        
        # Calculate term frequency
        word_counts = Counter(words)
        total_words = len(words)
        
        # Score words (simple TF * length bonus)
        scored = []
        for word, count in word_counts.items():
            tf = count / total_words
            length_bonus = min(len(word) / 10, 1.0)  # Longer words get bonus
            score = tf * (1 + length_bonus)
            scored.append((word, score))
        
        # Sort by score and return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [word for word, score in scored[:top_n]]
    
    def _extract_key_phrases(self, text: str, top_n: int = 10) -> list[str]:
        """
        Extract key phrases (2-3 word combinations).
        Uses a simplified RAKE-like approach.
        """
        # Split into sentences
        sentences = re.split(r'[.!?;]\s+', text)
        
        phrase_scores = Counter()
        
        for sentence in sentences:
            # Tokenize sentence
            words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{2,}\b', sentence.lower())
            
            # Generate 2-3 word phrases
            for i in range(len(words)):
                # Skip if starts with stop word
                if words[i] in self.STOP_WORDS:
                    continue
                
                # 2-word phrases
                if i + 1 < len(words) and words[i+1] not in self.STOP_WORDS:
                    phrase = f"{words[i]} {words[i+1]}"
                    phrase_scores[phrase] += 1
                
                # 3-word phrases
                if i + 2 < len(words):
                    if words[i+2] not in self.STOP_WORDS:
                        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                        phrase_scores[phrase] += 1
        
        # Return top phrases
        return [phrase for phrase, count in phrase_scores.most_common(top_n) if count > 1]


# Convenience function
def extract_entities(text: str) -> ExtractedEntities:
    """Extract entities from text using default extractor."""
    extractor = EntityExtractor()
    return extractor.extract(text)
