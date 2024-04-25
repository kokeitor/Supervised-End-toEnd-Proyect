from sklearn.utils import resample
from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE,SVMSMOTE,SMOTENC
from typing import Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder

def synthetic_resample(
                            X : np.ndarray,
                            y : np.ndarray,
                            ratio : float ,
                            technique : str = "SMOTE",
                            verbose : int = 0,
                            random_state : int = 1,
                        ) -> Tuple[np.ndarray,np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): 
        y (np.ndarray): y array should be labely codify previously : ["class0","class1",...] -> [0,1,2,...]
        ratio (float): ratio of resampling compared with more frequent class
        technique (str): technique for resampling : ["SMOTE", "oversampling","undersampling","ADASYN","BorderlineSMOTE","SVMSMOTE"]
        verbose (int): 1 fore extra information. Defaults to 0. 
        random_state (int): random state. Defaults to 1.

    Returns:
        Tuple[np.ndarray,np.ndarray]: --> x_resample, y_resample
    """
    # Checking types (or class) of input arguments X and y
    if not isinstance(X, (list,np.ndarray)) or not isinstance(y,(list,np.ndarray)):
        raise TypeError("X and/or y should be an array-like object (list or numpy array)")
    
    # Transforming list to arrays
    if isinstance(X, (list)):
        X = np.array(X)
    if isinstance(y,(list)): 
        y = np.array(y)
        
    # Check size of X and y
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y should have the same number of samples")
        
    if verbose == 1:
        print(f"y input type : {y.dtype}"," | y input shape : ",y.shape)
        print(f"X input type : {X.dtype}","| X input shape : ", X.shape)
    old_classes = np.unique(y)
    
    if not _is_numeric(y)[0]:
        print("-----------------------------------------------------------")
        print(f"ValueError : y contains {y.dtype} data types and y array should contain number data types\nNote: A label encoding will be applied to y")
        y = LabelEncoder().fit_transform(y)
        if verbose == 1:
            print(f"y new type : {y.dtype}")
            print(f"y old classes : {old_classes}")
            print(f"y new codify classes : {np.unique(y)}")
        
    if not _is_numeric(X)[0]:
        raise ValueError(f"X {_is_numeric(X)[1]}")

        
    if verbose == 1:
        print("-----------------------------------------------------------")
        print(f"Original dataset number of samples : {X.shape[0]}")
        print("Classes in the target variable : ", np.unique(y))
        print(f"Class frequencies : {np.bincount(y)}")
        for idx, v in enumerate(np.bincount(y)):
            print(f'Proportion of class {idx} : {100*v/y.shape[0] : .2f}','%')

    # Calculate min class label, max class label and new samples for resample
    min_class = np.argmin(np.bincount(y))
    max_class = np.argmax(np.bincount(y))
    new_class_samples = int(ratio*(np.bincount(y)[max_class]))
    
    # Dictionary for SMOTE resampling
    smote_class_samples = {}
    for class_,class_samples in enumerate(np.bincount(y)):
        if class_ == min_class:
            smote_class_samples[class_] = new_class_samples
        else:
            smote_class_samples[class_] = class_samples
        
    # OPTION RESAMPLING DICTIONARY:
    resampling_options = {
                            "SMOTE": SMOTE(
                                            sampling_strategy = smote_class_samples,
                                            random_state=random_state,
                                            k_neighbors=5,
                                            ), 
                            "oversampling": None, 
                            "undersampling" :None,
                            "ADASYN": ADASYN(
                                            sampling_strategy = smote_class_samples,
                                            random_state=random_state,
                                            n_neighbors=5,
                                            ), 
                            "BorderlineSMOTE": BorderlineSMOTE(
                                                                sampling_strategy = smote_class_samples,
                                                                random_state=random_state,
                                                                k_neighbors=5,
                                                                m_neighbors = 10,
                                                                kind = "borderline-1"
                                                                ), 
                            "SVMSMOTE": SVMSMOTE(
                                                sampling_strategy = smote_class_samples,
                                                random_state=random_state,
                                                k_neighbors=5,
                                                m_neighbors = 10,
                                                out_step = 0.5
                                            ), 
                            
                        }
    # Resampling
    if (tool := resampling_options.get(technique)) !=  None:
        if verbose ==1:
            print("-----------------------------------------------------------")
            print(F"Using: {technique} resampling technique")
        X_resampled, y_resampled= tool.fit_resample(X, y)
    elif (tool := resampling_options.get(technique)) == None and technique == "oversampling":
        X_resampled, y_resampled = resample(
                                                X[y == min_class],
                                                y[y == min_class],
                                                replace = True,
                                                n_samples = new_class_samples,
                                                random_state=random_state
                                            )
        X_resampled = np.vstack((X[y != min_class],X_resampled))
        y_resampled = np.hstack((y[y!= min_class],y_resampled))
        
    elif (tool := resampling_options.get(technique)) == None and technique == "undersampling":
        X_resampled, y_resampled = resample(
                                                X[y == max_class],
                                                y[y == max_class],
                                                replace = False,
                                                n_samples = new_class_samples,
                                                random_state = random_state
                                            )
        X_resampled = np.vstack((X[y != max_class],X_resampled))
        y_resampled = np.hstack((y[y != max_class],y_resampled))
        
    else:
        raise ValueError(f"{technique} is not a valid resampling technique")
    
    if verbose == 1:
        print("-----------------------------------------------------------")
        print("X resampled shape : ", X_resampled.shape)
        print("y resampled shape : ", y_resampled.shape)
        print(f"New dataset number of samples : {X_resampled.shape[0]}")
        print(f"% of increment compare to original dataset : {100*(X_resampled.shape[0]-X.shape[0])/X.shape[0]:.2f} %")
        print("Target classes : ", np.unique(y_resampled))
        print(f"New class frequencies : {np.bincount(y_resampled)}")
        for idx, v in enumerate(np.bincount(y_resampled)):
            print(f'New proportion of class {idx} : {100*v/y_resampled.shape[0] : .2f}','%')
            
    return X_resampled, y_resampled
    

def _is_numeric(X : np.ndarray) -> Tuple[bool,str]:
    """_summary_

    Args:
        X (np.ndarray): _description_

    Returns:
        Tuple[bool,str]: _description_
    """
    # Flatten the array to handle multi-dimensional arrays
    X_flat = X.flatten()
    
    # Check if the flattened array contains integers
    if np.issubdtype(X_flat.dtype, np.integer):
        return True, "contains integer data types"
    # Check if the flattened array contains floats
    elif np.issubdtype(X_flat.dtype, np.floating):
        return True, "contains float data types"
    # Check if the flattened array contains a mix of integers and floats
    elif np.issubdtype(X_flat.dtype, np.number):
        return True, "contains a mix of integer and float data types"
    else:
        return False, "does not contain numeric data type"
        
        
        
def testing() -> None:
    y =  ["A","B","B","B","B"]*30
    x = np.random.rand(len(y),3)
    """    x_new , y_new = synthetic_resample(
                            X = x,
                            y  = y,
                            ratio = 0.5 ,
                            technique = "SMOTE",
                            verbose  = 1
                            )"""
    x_new , y_new = synthetic_resample(
                            X = x,
                            y  = y,
                            ratio = 0.5 ,
                            technique = "ADASYN",
                            verbose  = 1
                            )
    x_new , y_new = synthetic_resample(
                        X = x,
                        y  = y,
                        ratio = 0.5 ,
                        technique = "SVMSMOTE",
                        verbose  = 1
                        )
    x_new , y_new = synthetic_resample(
                    X = x,
                    y  = y,
                    ratio = 0.5 ,
                    technique = "BorderlineSMOTE",
                    verbose  = 1
                    )
if __name__ == "__main__":
    testing()