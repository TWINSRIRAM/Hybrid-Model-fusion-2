import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch
import os
# Define attack categories
ATTACK_CATEGORIES_19 = { 
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_6 = {  
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_2 = {  
    'ARP_Spoofing': 'attack',
    'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack',
    'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack',
    'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack',
    'Recon-Ping_Sweep': 'attack',
    'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack',
    'TCP_IP-DDoS-ICMP': 'attack',
    'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack',
    'TCP_IP-DDoS-UDP': 'attack',
    'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack',
    'TCP_IP-DoS-TCP': 'attack',
    'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}

def get_attack_category(file_name, class_config):
    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories = ATTACK_CATEGORIES_6
    else:
        categories = ATTACK_CATEGORIES_19

    for key in categories:
        if key in file_name:
            return categories[key]

def load_and_preprocess_data(data_dir, class_config):
    train_files = [f"{data_dir}/train/{f}" for f in os.listdir(f"{data_dir}/train") if f.endswith('.csv')]
    test_files = [f"{data_dir}/test/{f}" for f in os.listdir(f"{data_dir}/test") if f.endswith('.csv')]

    train_df = pd.concat([pd.read_csv(f).assign(file=f) for f in train_files], ignore_index=True)
    test_df = pd.concat([pd.read_csv(f).assign(file=f) for f in test_files], ignore_index=True)

    train_df['Attack_Type'] = train_df['file'].apply(lambda x: get_attack_category(x, class_config))
    test_df['Attack_Type'] = test_df['file'].apply(lambda x: get_attack_category(x, class_config))

    X_train = train_df.drop(['Attack_Type', 'file'], axis=1)
    y_train = train_df['Attack_Type']
    X_test = test_df.drop(['Attack_Type', 'file'], axis=1)
    y_test = test_df['Attack_Type']

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
        X_train, y_train_encoded, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape and convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    return X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor, label_encoder
