"""
Simple Column Mapper
Static mapping from common CIC-IDS2017 column name variations to standard names
No fuzzy matching - just direct dictionary lookup
"""
from typing import Dict, List, Tuple
import pandas as pd


class SimpleColumnMapper:
    """
    Simple static column mapper using dictionary lookup
    Maps common CIC-IDS2017 column name variations to standard format
    """
    
    # Direct mapping: source -> target
    # Mencakup variasi umum dari CIC-IDS2017
    COLUMN_MAPPING = {
        # === Destination Port ===
        "dst port": "Destination Port",
        "dstport": "Destination Port",
        "destination port": "Destination Port",
        
        # === Flow Duration ===
        "flow duration": "Flow Duration",
        
        # === Packet Counts ===
        "tot fwd pkts": "Total Fwd Packets",
        "total fwd packets": "Total Fwd Packets",
        
        "totlen fwd pkts": "Total Length of Fwd Packets",
        "total length of fwd packets": "Total Length of Fwd Packets",
        
        # === Fwd Packet Length ===
        "fwd pkt len max": "Fwd Packet Length Max",
        "fwd packet length max": "Fwd Packet Length Max",
        
        "fwd pkt len min": "Fwd Packet Length Min",
        "fwd packet length min": "Fwd Packet Length Min",
        
        "fwd pkt len mean": "Fwd Packet Length Mean",
        "fwd packet length mean": "Fwd Packet Length Mean",
        
        "fwd pkt len std": "Fwd Packet Length Std",
        "fwd packet length std": "Fwd Packet Length Std",
        
        # === Bwd Packet Length ===
        "bwd pkt len max": "Bwd Packet Length Max",
        "bwd packet length max": "Bwd Packet Length Max",
        
        "bwd pkt len min": "Bwd Packet Length Min",
        "bwd packet length min": "Bwd Packet Length Min",
        
        "bwd pkt len mean": "Bwd Packet Length Mean",
        "bwd packet length mean": "Bwd Packet Length Mean",
        
        # === Flow Rate ===
        "flow byts/s": "Flow Bytes/s",
        "flow bytes/s": "Flow Bytes/s",
        
        "flow pkts/s": "Flow Packets/s",
        "flow packets/s": "Flow Packets/s",
        
        # === Flow IAT ===
        "flow iat mean": "Flow IAT Mean",
        "flow iat std": "Flow IAT Std",
        "flow iat max": "Flow IAT Max",
        "flow iat min": "Flow IAT Min",
        
        # === Fwd IAT ===
        "fwd iat mean": "Fwd IAT Mean",
        "fwd iat std": "Fwd IAT Std",
        "fwd iat min": "Fwd IAT Min",
        
        # === Bwd IAT ===
        "bwd iat tot": "Bwd IAT Total",
        "bwd iat total": "Bwd IAT Total",
        "bwd iat mean": "Bwd IAT Mean",
        "bwd iat std": "Bwd IAT Std",
        "bwd iat max": "Bwd IAT Max",
        "bwd iat min": "Bwd IAT Min",
        
        # === Flags ===
        "fwd psh flags": "Fwd PSH Flags",
        "fwd urg flags": "Fwd URG Flags",
        
        # === Header Length ===
        "fwd header len": "Fwd Header Length",
        "fwd header length": "Fwd Header Length",
        
        "bwd header len": "Bwd Header Length",
        "bwd header length": "Bwd Header Length",
        
        # === Bwd Rate ===
        "bwd pkts/s": "Bwd Packets/s",
        "bwd packets/s": "Bwd Packets/s",
        
        # === Packet Length Stats ===
        "pkt len min": "Min Packet Length",
        "min packet length": "Min Packet Length",
        
        "pkt len max": "Max Packet Length",
        "max packet length": "Max Packet Length",
        
        "pkt len mean": "Packet Length Mean",
        "packet length mean": "Packet Length Mean",
        
        "pkt len var": "Packet Length Variance",
        "packet length variance": "Packet Length Variance",
        
        # === Flag Counts ===
        "fin flag cnt": "FIN Flag Count",
        "fin flag count": "FIN Flag Count",
        
        "rst flag cnt": "RST Flag Count",
        "rst flag count": "RST Flag Count",
        
        "psh flag cnt": "PSH Flag Count",
        "psh flag count": "PSH Flag Count",
        
        "ack flag cnt": "ACK Flag Count",
        "ack flag count": "ACK Flag Count",
        
        "urg flag cnt": "URG Flag Count",
        "urg flag count": "URG Flag Count",
        
        # === Ratio ===
        "down/up ratio": "Down/Up Ratio",
        
        # === Init Window ===
        "init fwd win byts": "Init_Win_bytes_forward",
        "init_win_bytes_forward": "Init_Win_bytes_forward",
        "init win bytes forward": "Init_Win_bytes_forward",
        
        "init bwd win byts": "Init_Win_bytes_backward",
        "init_win_bytes_backward": "Init_Win_bytes_backward",
        "init win bytes backward": "Init_Win_bytes_backward",
        
        # === Active Data ===
        "fwd act data pkts": "act_data_pkt_fwd",
        "act_data_pkt_fwd": "act_data_pkt_fwd",
        
        # === Segment Size ===
        "fwd seg size min": "min_seg_size_forward",
        "min_seg_size_forward": "min_seg_size_forward",
        
        # === Active Stats ===
        "active mean": "Active Mean",
        "active std": "Active Std",
        "active max": "Active Max",
        "active min": "Active Min",
        
        # === Idle Stats ===
        "idle mean": "Idle Mean",
        "idle std": "Idle Std",
    }
    
    @classmethod
    def map_columns(cls, df: pd.DataFrame, required_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Map DataFrame columns to standard names
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            (mapped_df, mapping_dict)
        """
        # Strip whitespace and create lowercase version for matching
        df. columns = df.columns.str. strip()
        
        mapping = {}
        
        for col in df.columns:
            col_lower = col.lower(). strip()
            
            # Check if already matches a required column (case-insensitive)
            for req_col in required_columns:
                if col_lower == req_col.lower():
                    if col != req_col:  # Only map if different
                        mapping[col] = req_col
                    break
            else:
                # Check in our static mapping
                if col_lower in cls.COLUMN_MAPPING:
                    target = cls.COLUMN_MAPPING[col_lower]
                    if target in required_columns:
                        mapping[col] = target
        
        # Apply mapping
        df_mapped = df.rename(columns=mapping)
        
        return df_mapped, mapping
    
    @classmethod
    def get_mapping_report(cls, source_columns: List[str], required_columns: List[str]) -> Dict:
        """
        Get report of how columns will be mapped
        """
        # Create temporary DataFrame just for mapping check
        temp_df = pd. DataFrame(columns=source_columns)
        mapped_df, mapping = cls.map_columns(temp_df, required_columns)
        
        mapped_targets = set(mapping.values())
        found_columns = []
        
        # Check which required columns are available after mapping
        for req_col in required_columns:
            # Direct match
            if req_col in mapped_df.columns:
                found_columns.append(req_col)
            # Or was mapped
            elif req_col in mapped_targets:
                found_columns. append(req_col)
            # Case-insensitive check
            else:
                for col in mapped_df.columns:
                    if col.lower() == req_col.lower():
                        found_columns.append(req_col)
                        break
        
        missing = [col for col in required_columns if col not in found_columns]
        
        return {
            "mapping": mapping,
            "mapped_count": len(found_columns),
            "total_required": len(required_columns),
            "missing_columns": missing,
            "is_complete": len(missing) == 0
        }


# Simple function for direct use
def map_dataframe_columns(df: pd. DataFrame, required_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Convenience function to map DataFrame columns"""
    return SimpleColumnMapper.map_columns(df, required_columns)