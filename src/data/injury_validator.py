"""
Injury Data Validation Module

Provides validation layer to ensure injury data quality before use in predictions.

Features:
- Required field validation
- Status value validation
- Data freshness checks
- Quality metric logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float  # 0-100, higher is better
    record_count: int
    valid_record_count: int
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'quality_score': self.quality_score,
            'record_count': self.record_count,
            'valid_record_count': self.valid_record_count,
        }


class InjuryDataValidator:
    """
    Validates injury data for quality and completeness.
    
    Checks:
    1. Required fields are present
    2. Status values are valid
    3. Data is not stale
    4. No duplicate entries
    5. Position values are valid
    """
    
    # Required fields for injury data
    REQUIRED_FIELDS = ['player_name', 'status', 'team']
    
    # Valid injury status values
    VALID_STATUSES = {
        'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'PROBABLE',
        'IR', 'PUP', 'SUSPENDED', 'NFI',
        # Lowercase variants
        'out', 'doubtful', 'questionable', 'probable',
        'ir', 'pup', 'suspended', 'nfi',
        # Mixed case
        'Out', 'Doubtful', 'Questionable', 'Probable'
    }
    
    # Valid positions
    VALID_POSITIONS = {'QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'S', 'CB'}
    
    # NFL teams
    VALID_TEAMS = {
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS',
        # Full names and variants
        'Arizona', 'Atlanta', 'Baltimore', 'Buffalo', 'Carolina',
        'Chicago', 'Cincinnati', 'Cleveland', 'Dallas', 'Denver',
        'Detroit', 'Green Bay', 'Houston', 'Indianapolis', 'Jacksonville',
        'Kansas City', 'Las Vegas', 'Los Angeles', 'Miami', 'Minnesota',
        'New England', 'New Orleans', 'New York', 'Philadelphia', 'Pittsburgh',
        'San Francisco', 'Seattle', 'Tampa Bay', 'Tennessee', 'Washington',
        # Old abbreviations
        'OAK', 'SD', 'STL', 'LA'
    }
    
    def __init__(self, max_data_age_hours: int = 48):
        """
        Initialize validator.
        
        Args:
            max_data_age_hours: Maximum acceptable age of data in hours
        """
        self.max_data_age_hours = max_data_age_hours
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate injury DataFrame.
        
        Args:
            df: DataFrame with injury data
            
        Returns:
            ValidationResult with validation outcome and quality metrics
        """
        errors = []
        warnings = []
        
        if df.empty:
            return ValidationResult(
                is_valid=False,
                errors=['Empty DataFrame'],
                warnings=[],
                quality_score=0.0,
                record_count=0,
                valid_record_count=0
            )
        
        record_count = len(df)
        
        # Check required fields
        field_errors = self._validate_required_fields(df)
        errors.extend(field_errors)
        
        # Check status values
        status_warnings = self._validate_status_values(df)
        warnings.extend(status_warnings)
        
        # Check data freshness
        freshness_warnings = self._validate_freshness(df)
        warnings.extend(freshness_warnings)
        
        # Check for duplicates
        dup_warnings = self._validate_duplicates(df)
        warnings.extend(dup_warnings)
        
        # Check position values
        pos_warnings = self._validate_positions(df)
        warnings.extend(pos_warnings)
        
        # Check team values
        team_warnings = self._validate_teams(df)
        warnings.extend(team_warnings)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, errors, warnings)
        
        # Count valid records
        valid_record_count = self._count_valid_records(df)
        
        is_valid = len(errors) == 0
        
        # Log results
        if errors:
            logger.warning(f"Injury data validation errors: {errors}")
        if warnings:
            logger.info(f"Injury data validation warnings: {warnings}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            record_count=record_count,
            valid_record_count=valid_record_count
        )
    
    def _validate_required_fields(self, df: pd.DataFrame) -> List[str]:
        """Check that required fields are present and non-null."""
        errors = []
        
        for field in self.REQUIRED_FIELDS:
            if field not in df.columns:
                errors.append(f"Missing required field: {field}")
            elif df[field].isna().all():
                errors.append(f"Required field '{field}' is all null")
        
        return errors
    
    def _validate_status_values(self, df: pd.DataFrame) -> List[str]:
        """Check that status values are valid."""
        warnings = []
        
        if 'status' not in df.columns:
            return warnings
        
        invalid_statuses = df[~df['status'].isin(self.VALID_STATUSES)]['status'].unique()
        
        if len(invalid_statuses) > 0:
            warnings.append(f"Unknown status values: {list(invalid_statuses)}")
        
        return warnings
    
    def _validate_freshness(self, df: pd.DataFrame) -> List[str]:
        """Check that data is not stale."""
        warnings = []
        
        if 'fetched_at' not in df.columns:
            warnings.append("No 'fetched_at' timestamp - cannot verify data freshness")
            return warnings
        
        try:
            # Handle both datetime and string formats
            if df['fetched_at'].dtype == 'object':
                df['fetched_at'] = pd.to_datetime(df['fetched_at'])
            
            oldest = df['fetched_at'].min()
            age_hours = (datetime.now() - oldest).total_seconds() / 3600
            
            if age_hours > self.max_data_age_hours:
                warnings.append(f"Data is {age_hours:.1f} hours old (max: {self.max_data_age_hours})")
        except Exception as e:
            warnings.append(f"Could not parse fetched_at timestamp: {e}")
        
        return warnings
    
    def _validate_duplicates(self, df: pd.DataFrame) -> List[str]:
        """Check for duplicate player entries."""
        warnings = []
        
        if 'player_name' not in df.columns:
            return warnings
        
        duplicates = df['player_name'].duplicated().sum()
        
        if duplicates > 0:
            warnings.append(f"{duplicates} duplicate player entries found")
        
        return warnings
    
    def _validate_positions(self, df: pd.DataFrame) -> List[str]:
        """Check that position values are valid."""
        warnings = []
        
        if 'position' not in df.columns:
            warnings.append("No 'position' field in data")
            return warnings
        
        # Check for fantasy-relevant positions only
        fantasy_positions = {'QB', 'RB', 'WR', 'TE'}
        position_counts = df['position'].value_counts()
        
        for pos in fantasy_positions:
            if pos not in position_counts.index:
                warnings.append(f"No {pos} players in injury data")
        
        return warnings
    
    def _validate_teams(self, df: pd.DataFrame) -> List[str]:
        """Check that team values are valid."""
        warnings = []
        
        if 'team' not in df.columns:
            return warnings
        
        unknown_teams = df[~df['team'].isin(self.VALID_TEAMS)]['team'].unique()
        
        if len(unknown_teams) > 0:
            warnings.append(f"Unknown team values: {list(unknown_teams)[:5]}")  # Limit to 5
        
        return warnings
    
    def _calculate_quality_score(
        self, 
        df: pd.DataFrame, 
        errors: List[str], 
        warnings: List[str]
    ) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Scoring:
        - Base score: 100
        - Each error: -25 points
        - Each warning: -5 points
        - Missing optional fields: -2 points each
        - High null percentage: -10 points
        """
        score = 100.0
        
        # Penalty for errors
        score -= len(errors) * 25
        
        # Penalty for warnings
        score -= len(warnings) * 5
        
        # Penalty for missing optional fields
        optional_fields = ['injury_type', 'confidence', 'position']
        for field in optional_fields:
            if field not in df.columns:
                score -= 2
        
        # Penalty for high null percentage in required fields
        for field in self.REQUIRED_FIELDS:
            if field in df.columns:
                null_pct = df[field].isna().mean()
                if null_pct > 0.1:  # More than 10% nulls
                    score -= 10 * null_pct
        
        return max(0, min(100, score))
    
    def _count_valid_records(self, df: pd.DataFrame) -> int:
        """Count records that pass all validation checks."""
        if df.empty:
            return 0
        
        valid_mask = pd.Series([True] * len(df))
        
        # Must have player_name
        if 'player_name' in df.columns:
            valid_mask &= df['player_name'].notna()
        
        # Must have valid status
        if 'status' in df.columns:
            valid_mask &= df['status'].isin(self.VALID_STATUSES)
        
        # Must have team
        if 'team' in df.columns:
            valid_mask &= df['team'].notna()
        
        return valid_mask.sum()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize injury data.
        
        - Standardizes status values to uppercase
        - Fills missing confidence with default
        - Removes duplicate entries (keeps highest confidence)
        
        Args:
            df: Raw injury DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Standardize status to uppercase
        if 'status' in df.columns:
            df['status'] = df['status'].str.upper()
        
        # Fill missing confidence
        if 'confidence' not in df.columns:
            df['confidence'] = 0.7  # Default confidence
        else:
            df['confidence'] = df['confidence'].fillna(0.7)
        
        # Remove duplicates, keeping highest confidence
        if 'player_name' in df.columns:
            df = df.sort_values('confidence', ascending=False)
            df = df.drop_duplicates(subset=['player_name'], keep='first')
        
        # Ensure required columns exist
        for col in ['injury_type', 'position', 'team']:
            if col not in df.columns:
                df[col] = None
        
        return df.reset_index(drop=True)


def validate_injury_data(df: pd.DataFrame) -> ValidationResult:
    """
    Convenience function to validate injury data.
    
    Args:
        df: Injury DataFrame to validate
        
    Returns:
        ValidationResult with validation outcome
    """
    validator = InjuryDataValidator()
    return validator.validate(df)


def clean_injury_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to clean injury data.
    
    Args:
        df: Injury DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    validator = InjuryDataValidator()
    return validator.clean_data(df)
