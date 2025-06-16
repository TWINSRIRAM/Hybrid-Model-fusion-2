import pandas as pd
import os
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------- Attack Category Mappings ---------------------- #

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

# ---------------------- Helper Function ---------------------- #

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
    return 'Unknown'

def load_and_preprocess_data(class_config):
    train_files = [os.path.join("data", "train", f) for f in os.listdir(os.path.join("data", "train")) if f.endswith(".csv")]
    test_files = [os.path.join("data", "test", f) for f in os.listdir(os.path.join("data", "test")) if f.endswith(".csv")]

    train_df = pd.concat([pd.read_csv(f).assign(file=f) for f in train_files], ignore_index=True)
    test_df = pd.concat([pd.read_csv(f).assign(file=f) for f in test_files], ignore_index=True)

    test_sample = test_df.sample(frac=0.2, random_state=42)
    train_sample = train_df.sample(frac=0.8, random_state=42)
    valtest_df = pd.concat([test_sample, train_sample], ignore_index=True)

    train_df["Attack_Type"] = train_df["file"].apply(lambda x: get_attack_category(x, class_config))
    valtest_df["Attack_Type"] = valtest_df["file"].apply(lambda x: get_attack_category(x, class_config))

    X_train = train_df.drop(columns=["Attack_Type", "file"])
    y_train = train_df["Attack_Type"]
    X_valtest = valtest_df.drop(columns=["Attack_Type", "file"])
    y_valtest = valtest_df["Attack_Type"]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valtest_encoded = label_encoder.transform(y_valtest)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valtest_scaled = scaler.transform(X_valtest)

    X_val, X_test, y_val, y_test = train_test_split(X_valtest_scaled, y_valtest_encoded, test_size=0.5, stratify=y_valtest_encoded, random_state=42)

    return (
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train_encoded, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
        label_encoder
    )
