"""
Transaction Data Generator for Fraud Detection System
Generates realistic financial transactions with both normal and fraudulent patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


class TransactionGenerator:
    """Generates realistic financial transaction data with fraud patterns"""
    
    def __init__(self, n_transactions=10000, fraud_ratio=0.02):
        """
        Initialize transaction generator
        
        Args:
            n_transactions (int): Total number of transactions to generate
            fraud_ratio (float): Proportion of fraudulent transactions (default 2%)
        """
        self.n_transactions = n_transactions
        self.fraud_ratio = fraud_ratio
        self.n_frauds = int(n_transactions * fraud_ratio)
        self.n_normal = n_transactions - self.n_frauds
        
        # Merchant categories
        self.merchant_categories = [
            'grocery', 'restaurant', 'gas_station', 'online_retail', 
            'electronics', 'pharmacy', 'entertainment', 'travel',
            'subscription', 'utilities', 'clothing', 'home_improvement'
        ]
        
        # Location data (major cities)
        self.locations = [
            'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
            'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA',
            'Dallas, TX', 'San Jose, CA', 'Austin, TX', 'Jacksonville, FL',
            'Miami, FL', 'Seattle, WA', 'Boston, MA', 'Denver, CO'
        ]
        
    def generate_normal_transactions(self):
        """Generate normal transaction patterns"""
        transactions = []
        
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(self.n_normal):
            # Normal transaction patterns
            hour = np.random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
                                   p=[0.05, 0.08, 0.1, 0.12, 0.15, 0.12, 0.1, 0.08, 0.08, 0.05, 0.04, 0.02, 0.01])
            
            transaction_time = base_date + timedelta(
                days=int(random.randint(0, 29)),
                hours=int(hour),
                minutes=int(random.randint(0, 59)),
                seconds=int(random.randint(0, 59))
            )
            
            # Normal amount distribution by category
            category = random.choice(self.merchant_categories)
            
            if category == 'grocery':
                amount = np.random.gamma(shape=2, scale=25)
            elif category == 'restaurant':
                amount = np.random.gamma(shape=2, scale=15)
            elif category == 'gas_station':
                amount = np.random.normal(45, 15)
            elif category == 'online_retail':
                amount = np.random.gamma(shape=1.5, scale=30)
            elif category == 'electronics':
                amount = np.random.gamma(shape=1.2, scale=150)
            elif category == 'travel':
                amount = np.random.gamma(shape=1.5, scale=200)
            else:
                amount = np.random.gamma(shape=2, scale=20)
            
            amount = max(1.0, min(5000.0, amount))  # Cap between $1 and $5000
            
            # Card age (in days) - normal cards are 180-1800 days old
            card_age = random.randint(180, 1800)
            
            # Transaction frequency (transactions per day in last 7 days)
            freq = np.random.poisson(lam=2.5)
            
            # Location (usually consistent)
            location = random.choice(self.locations[:10])  # Normal users stick to fewer locations
            
            transactions.append({
                'transaction_id': f'TXN{i:06d}',
                'timestamp': transaction_time,
                'amount': round(amount, 2),
                'merchant_id': f'MERCH_{category.upper()}_{random.randint(1000, 9999)}',
                'merchant_category': category,
                'location': location,
                'card_age': card_age,
                'transaction_frequency': freq,
                'is_fraud': 0
            })
        
        return transactions
    
    def generate_fraudulent_transactions(self):
        """Generate fraudulent transaction patterns"""
        transactions = []
        
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(self.n_frauds):
            # Fraud patterns: unusual hours, high amounts, rapid sequences
            
            # Fraud type selection
            fraud_type = random.choice(['high_amount', 'rapid_fire', 'unusual_location', 'new_card'])
            
            if fraud_type == 'high_amount':
                # High amount fraud
                amount = np.random.uniform(2000, 9999)
                hour = random.choice([0, 1, 2, 3, 4, 5, 22, 23])  # Unusual hours
                freq = np.random.randint(1, 4)
                card_age = random.randint(180, 1000)
                
            elif fraud_type == 'rapid_fire':
                # Multiple transactions in short time
                amount = np.random.uniform(300, 1500)
                hour = random.randint(0, 23)
                freq = np.random.randint(10, 25)  # Very high frequency
                card_age = random.randint(180, 1000)
                
            elif fraud_type == 'unusual_location':
                # Foreign/unusual location
                amount = np.random.uniform(500, 3000)
                hour = random.randint(0, 23)
                freq = np.random.randint(5, 15)
                card_age = random.randint(200, 1200)
                
            else:  # new_card
                # New card with high amount
                amount = np.random.uniform(1000, 5000)
                hour = random.choice([0, 1, 2, 3, 22, 23])
                freq = np.random.randint(3, 8)
                card_age = random.randint(1, 30)  # Very new card
            
            transaction_time = base_date + timedelta(
                days=int(random.randint(0, 29)),
                hours=int(hour),
                minutes=int(random.randint(0, 59)),
                seconds=int(random.randint(0, 59))
            )
            
            category = random.choice(['electronics', 'online_retail', 'travel', 'jewelry'])
            location = random.choice(self.locations)  # More random locations
            
            transactions.append({
                'transaction_id': f'TXN{self.n_normal + i:06d}',
                'timestamp': transaction_time,
                'amount': round(amount, 2),
                'merchant_id': f'MERCH_{category.upper()}_{random.randint(1000, 9999)}',
                'merchant_category': category,
                'location': location,
                'card_age': card_age,
                'transaction_frequency': freq,
                'is_fraud': 1
            })
        
        return transactions
    
    def generate_dataset(self):
        """Generate complete dataset with normal and fraudulent transactions"""
        print("Generating normal transactions...")
        normal_txns = self.generate_normal_transactions()
        
        print("Generating fraudulent transactions...")
        fraud_txns = self.generate_fraudulent_transactions()
        
        # Combine and shuffle
        all_transactions = normal_txns + fraud_txns
        random.shuffle(all_transactions)
        
        # Create DataFrame
        df = pd.DataFrame(all_transactions)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add additional features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print(f"\nDataset generated successfully!")
        print(f"Total transactions: {len(df)}")
        print(f"Normal transactions: {len(df[df['is_fraud'] == 0])}")
        print(f"Fraudulent transactions: {len(df[df['is_fraud'] == 1])}")
        print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
        
        return df


if __name__ == "__main__":
    # Generate dataset
    generator = TransactionGenerator(n_transactions=10000, fraud_ratio=0.02)
    df = generator.generate_dataset()
    
    # Save to CSV
    output_path = 'transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Display statistics
    print("\n=== Dataset Statistics ===")
    print(df.describe())
    print("\n=== Fraud Distribution ===")
    print(df['is_fraud'].value_counts())
    print("\n=== Sample Transactions ===")
    print(df.head(10))
