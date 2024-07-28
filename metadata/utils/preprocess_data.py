import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import os
from sklearn.model_selection import train_test_split


def read_csv(config):
        df_train_2024 = pd.read_csv(os.path.join(config["metadata_folder"],'train-metadata.csv'))
        df_test_2024 = pd.read_csv(os.path.join(config["metadata_folder"],'test-metadata.csv'))
        
        remove_columns = [col for col in df_train_2024.columns if (col not in df_test_2024.columns and col != "target")]
        df_train_2024 = df_train_2024.drop(columns=remove_columns, axis=1)
       
        num_cols = [
        'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 
        'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
        'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
        'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
        'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
        'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
        ]


        df_train_2024[num_cols] = df_train_2024[num_cols].fillna(df_train_2024[num_cols].median())
        df_train, new_num_cols, new_cat_cols = feature_engineering(df_train_2024.copy())
        df_test, _, _ = feature_engineering(df_test_2024.copy())

        util_cols = ['isic_id', 'patient_id', 'target']
        util_cols = util_cols + ["image_path"] if "image_path" in df_train_2024.columns else util_cols
        num_cols += new_num_cols
        # anatom_site_general
        cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"] + new_cat_cols
        train_cols =  cat_cols + num_cols + util_cols
        df_train  = df_train[train_cols]
        df_test = df_test[cat_cols + num_cols + util_cols[:-1]]

        return df_train, df_test, num_cols, cat_cols, util_cols

def preprocess(config):
        df_train, df_test, num_cols, cat_cols, utils_col = read_csv(config)
        meta_handler = MetaDataClass(df_train)
        
        category_encoder = OrdinalEncoder(categories='auto',dtype=int,handle_unknown='use_encoded_value',unknown_value=-2,encoded_missing_value=-1)
        onhot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_encoders = {'category_encoder': category_encoder, 'one_hot_encoder': onhot_encoder}

        minmax_scale = MinMaxScaler()
        num_encoders = {'minmax_scaler': minmax_scale}

        df_train = meta_handler.processCatAttr(df_train, cat_cols, 'train', cat_encoders, 'unknown')
        df_train = meta_handler.processNumAttr(df_train, num_cols, 'train', num_encoders, 'mode')
        df_test = meta_handler.processCatAttr(df_test, cat_cols, 'test')
        df_test = meta_handler.processNumAttr(df_test, num_cols, 'test')

        meta_features = [col for col in df_train.columns if col not in utils_col]
        df_train = meta_handler.split_data(df_train, config['n_folds'], True, 0.5)

        df_train_x = df_train[meta_features]
        df_train_y = df_train['target']

        df_train_x, df_valid_x, df_train_y, df_valid_y = train_test_split(df_train_x, df_train_y, test_size= config['valid_size'], stratify=df_train_y, random_state=42)

        df_test_x = df_test[meta_features]

        df = dict()
        df['x_train'] = df_train_x
        df['y_train'] = df_train_y
        df['x_valid'] = df_valid_x
        df['y_valid'] = df_valid_y
        df['x_test'] = df_test_x

        return df, meta_handler, meta_features

class MetaDataClass:
        def __init__(self, pd_frame):
                self.cat_pipeline = None
                self.num_pipeline = None
                self.num_fill_nan_value = 0
                self.cat_fill_nan_value = 'unknown'
        def processCatAttr(self, data, names, mode='train', encoders=None,fillnan_value= None):
                if(fillnan_value is not None):
                       self.num_fill_nan_value = fillnan_value

                data[names] = data[names].fillna(self.num_fill_nan_value)  # Missing value handling
                if mode == 'train':
                        self.cat_pipeline = Pipeline(steps=[(key, enc) for key, enc in encoders.items()])
                        new_df = self.cat_pipeline.fit_transform(data[names].astype('str'))
                elif mode == 'test' and self.cat_pipeline is not None:
                        new_df = self.cat_pipeline.transform(data[names].astype('str'))
                else:
                        assert('Invalid mode, should be either train or test')
                        return
                try:
                    feature_names = self.cat_pipeline.named_steps['one_hot_encoder'].get_feature_names_out(names)
                    new_df = pd.DataFrame(new_df, columns=feature_names, index=data.index)
                except:
                    print("One-hot encoding failed, using default column names")
                    new_df = pd.DataFrame(new_df, columns=names, index=data.index)

                return data.drop(columns=names).join(new_df)

        def processNumAttr(self, data, names, mode='train', encoders=None, fillnan_value = None):
                if(fillnan_value is not None):
                       self.num_fill_nan_value = fillnan_value
                for name in names:
                        if self.num_fill_nan_value == 'mode':
                                try:
                                        fill_value = data[name].mode()[0]
                                except:
                                        fill_value = data[name].mean()
                        elif self.num_fill_nan_value =='min':
                                fill_value = data[name].min()
                        elif self.num_fill_nan_value =='max':
                                fill_value = data[name].max()
                        else:
                                fill_value = 0

                        data[name] = data[name].fillna(fill_value) 
                        # check if contain inf value, then remove that column
                        if np.isinf(data[name]).any():
                                data.drop(name, axis=1, inplace=True)
                                print(f'Dropped {name} column due to inf values')
                                names.remove(name)
                
                if mode == 'train':
                        self.num_pipeline = Pipeline(steps=[(key, enc) for key, enc in encoders.items()])
                        data[names] = self.num_pipeline.fit_transform(data[names].astype(np.float32))
                elif mode == 'test' and self.cat_pipeline is not None:
                        data[names] = self.num_pipeline.transform(data[names].astype(np.float32))
                else:
                        assert('Invalid mode, should be either train or test')
                        return data

                return data
        
        @staticmethod
        def split_data(df, n_folds = 5, subsample=False, subsample_ratio=0.5, group_col='patient_id'):

                # Ensure target and group columns are defined in your dataset
                target_col = "target"  # Replace with your actual target column name

                # Create stratified group k-fold
                gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

                # Subsampling for balancing classes (if enabled)
                if subsample:
                        df_pos = df[df[target_col] == 1]
                        df_neg = df[df[target_col] == 0]
                        df_neg = df_neg.sample(frac=subsample_ratio, random_state=42)
                        df = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

                # Assign fold numbers
                df["fold"] = -1
                for idx, (train_idx, val_idx) in enumerate(gkf.split(df, df[target_col], groups=df[group_col])):
                        df.loc[val_idx, "fold"] = idx
                        if idx == 0:
                                print(f"Fold {idx}, size={len(val_idx)}, n_patient_id= {len(np.unique(val_idx))}")
                                print(f"  Train: index={train_idx}")
                                print(f"         group={df.loc[train_idx, group_col].values}")
                                print(f"  Test:  index={val_idx}")
                                print(f"         group={df.loc[val_idx, group_col].values}")
                
                return df
        
        @staticmethod
        def get_fold_data(df, fold):
                return df[df["fold"] == fold]
        
                        
       
def feature_engineering(df):
    # New features to try...
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

    # Taken from: https://www.kaggle.com/code/dschettler8845/isic-detect-skin-cancer-let-s-learn-together
    df["color_variance_ratio"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    df["border_color_interaction"] = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]
    df["size_color_contrast_ratio"] = df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]
    df["age_normalized_nevi_confidence"] = df["tbp_lv_nevi_confidence"] / df["age_approx"]
    df["color_asymmetry_index"] = df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]
    df["3d_volume_approximation"] = df["tbp_lv_areaMM2"] * np.sqrt(df["tbp_lv_x"]**2 + df["tbp_lv_y"]**2 + df["tbp_lv_z"]**2)
    df["color_range"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs() + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs() + (df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()
    df["shape_color_consistency"] = df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]
    df["border_length_ratio"] = df["tbp_lv_perimeterMM"] / (2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi))
    df["age_size_symmetry_index"] = df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]
    # Until here.
    
    new_num_cols = [
        "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
        "luminance_contrast", "lesion_color_difference", "border_complexity",
        "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
        "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",

        "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index", 
        "shape_complexity_index", "color_contrast_index", "log_lesion_area",
        "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
        "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
        "symmetry_perimeter_interaction", "comprehensive_lesion_index",
        
        "color_variance_ratio", "border_color_interaction", "size_color_contrast_ratio",
        "age_normalized_nevi_confidence", "color_asymmetry_index", "3d_volume_approximation",
        "color_range", "shape_color_consistency", "border_length_ratio", "age_size_symmetry_index",
    ]
    new_cat_cols = ["combined_anatomical_site"]
    return df, new_num_cols, new_cat_cols
