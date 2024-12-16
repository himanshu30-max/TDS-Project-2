# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "numpy",
#   "seaborn",
#   "scikit-learn",
#   "scipy",
#   "requests",
#   "matplotlib",
#   "tabulate",
#   "wordcloud",
#   "tenacity"
# ]
# ///

"""
Imports
"""

from typing import Optional
from PIL import Image
import io
import os
import sys
import json
from typing import Tuple, Optional
from wordcloud import WordCloud
from tenacity import retry, stop_after_attempt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from scipy.stats import skew, kurtosis, chi2_contingency


AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Tries to load a CSV file with various encodings to handle different character encodings.

    :param file_path: Path to the CSV file.
    :return: Pandas DataFrame containing the dataset.
    """
    encodings_to_try = [
        'utf8', 'latin1', 'ascii', 'us-ascii', 'big5', 'big5-tw', 'csbig5', 'big5hkscs', 'big5-hkscs',
        'cp037', 'IBM037', 'IBM039', 'cp273', 'IBM273', 'cp424', 'EBCDIC-CP-HE', 'IBM424',
        'cp437', 'IBM437', 'cp500', 'EBCDIC-CP-BE', 'EBCDIC-CP-CH', 'IBM500', 'cp720',
        'cp737', 'cp775', 'IBM775', 'cp850', 'IBM850', 'cp852', 'IBM852', 'cp855', 'IBM855',
        'cp856', 'cp857', 'IBM857', 'cp858', 'IBM858', 'cp860', 'IBM860', 'cp861', 'IBM861',
        'cp862', 'IBM862', 'cp863', 'IBM863', 'cp864', 'IBM864', 'cp865', 'IBM865', 'cp866',
        'IBM866', 'cp869', 'IBM869', 'cp874', 'cp875', 'cp932', 'ms932', 'ms-kanji', 'windows-31j',
        'cp949', 'ms949', 'uhc', 'cp950', 'ms950', 'cp1006', 'cp1026', 'ibm1026', 'cp1125',
        'ibm1125', 'cp1140', 'ibm1140', 'cp1250', 'windows-1250', 'cp1251', 'windows-1251',
        'cp1252', 'windows-1252', 'cp1253', 'windows-1253', 'cp1254', 'windows-1254', 'cp1255',
        'windows-1255', 'cp1256', 'windows-1256', 'cp1257', 'windows-1257', 'cp1258', 'windows-1258',
        'euc_jp', 'eucjp', 'ujis', 'u-jis', 'euc_jis_2004', 'jisx0213', 'eucjis2004', 'euc_jisx0213',
        'eucjisx0213', 'euc_kr', 'euckr', 'korean', 'ks_c-5601', 'ks_c-5601-1987', 'ksx1001',
        'ks_x-1001', 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022jp', 'iso-2022-jp',
        'iso2022_jp_1', 'iso2022jp-1', 'iso2022_jp_2', 'iso2022jp-2', 'iso-2022-jp-2', 'iso2022_jp_2004',
        'iso2022jp-2004', 'iso-2022-jp-2004', 'iso2022_jp_3', 'iso2022jp-3', 'iso-2022-jp-3', 'iso2022_jp_ext',
        'iso2022jp-ext', 'iso-2022-jp-ext', 'iso2022_kr', 'iso2022kr', 'iso-2022-kr',
        'iso-8859-1', 'iso8859-1', '8859', 'cp819', 'latin', 'latin1', 'L1', 'iso8859_2', 'iso-8859-2',
        'latin2', 'L2', 'iso8859_3', 'iso-8859-3', 'latin3', 'L3', 'iso8859_4', 'iso-8859-4', 'latin4',
        'L4', 'iso8859_5', 'iso-8859-5', 'cyrillic', 'iso8859_6', 'iso-8859-6', 'arabic', 'iso8859_7',
        'iso-8859-7', 'greek', 'greek8', 'iso8859_8', 'iso-8859-8', 'hebrew', 'iso8859_9', 'iso-8859-9',
        'latin5', 'L5', 'iso8859_10', 'iso-8859-10', 'latin6', 'L6', 'iso8859_11', 'iso-8859-11', 'thai',
        'iso8859_13', 'iso-8859-13', 'latin7', 'L7', 'iso8859_14', 'iso-8859-14', 'latin8', 'L8', 'iso8859_15',
        'iso-8859-15', 'latin9', 'L9', 'iso8859_16', 'iso-8859-16', 'latin10', 'L10', 'johab', 'cp1361',
        'ms1361', 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'kz_1048', 'strk1048_2002', 'rk1048', 'mac_cyrillic',
        'maccyrillic', 'mac_greek', 'macgreek', 'mac_iceland', 'maciceland', 'mac_latin2', 'maclatin2',
        'maccentraleurope', 'mac_centeuro', 'mac_roman', 'macroman', 'macintosh', 'mac_turkish', 'macturkish',
        'ptcp154', 'csptcp154', 'pt154', 'cp154', 'cyrillic-asian', 'shift_jis', 'csshiftjis', 'shiftjis',
        'sjis', 's_jis', 'shift_jis_2004', 'shiftjis2004', 'sjis_2004', 'sjis2004', 'shift_jisx0213',
        'shiftjisx0213', 'sjisx0213', 's_jisx0213', 'utf_32', 'U32', 'utf32', 'utf_32_be', 'UTF-32BE',
        'utf_32_le', 'UTF-32LE', 'utf_16', 'U16', 'utf16', 'utf_16_be', 'UTF-16BE', 'utf_16_le', 'UTF-16LE',
        'utf_7', 'U7', 'unicode-1-1-utf-7', 'utf_8', 'U8', 'UTF', 'cp65001', 'utf_8_sig'
    ]

    for encoding in encodings_to_try:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded data with {encoding} encoding.")
            return data
        except UnicodeDecodeError:
            print(f"""Failed to load data with {
                  encoding} encoding. Trying next...""")
        except Exception as e:
            print(f"An error occurred with {encoding} encoding: {e}")
            continue

    # We can't continue if we reach here
    raise NotImplementedError(
        "Failed to load the CSV file with all attempted encodings.")


def summarize_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Summarizes the dataset by providing numerical and categorical summary statistics,
    and counting missing values.

    :param data: Pandas DataFrame containing the dataset.
    :return: A tuple containing:
             - Numerical summary (DataFrame)
             - Categorical summary (DataFrame)
             - Missing values (Series)
    """
    numerical_summary = data.describe().T[['mean', '50%', 'std', 'min', 'max']]
    categorical_summary = data.select_dtypes(
        include=['object']).describe().T[['top', 'freq']]
    missing_values = data.isnull().sum()

    return numerical_summary, categorical_summary, missing_values


def generate_visualizations(data: pd.DataFrame, file_path: str) -> str:
    """
    Generates visualizations (e.g., correlation heatmap) based on the dataset.

    """
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    corr_path = os.path.join(folder_path, "correlation_matrix.png")
    plt.figure(figsize=(10, 8))
    numerical_cols = data.select_dtypes(include=['number']).columns
    if numerical_cols.shape[0] > 1:
        correlation_matrix = data[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig(corr_path)
        plt.close()

    return corr_path


@retry(stop=stop_after_attempt(3))
def analyze_with_llm(filename: str, api_key: str) -> Optional[str]:
    """
    Analyzes the dataset using an LLM via a proxy API and returns string in markdown format.

    """

    data = load_data(filename)
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(
        include=['object', 'category']).columns.tolist()
    time_series_cols = [
        col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    text_cols = [
        col for col in categorical_cols if data[col].str.len().mean() > 50]
    column_info = {col: str(data[col].dtype) for col in data.columns}

    dataset_summary = data_summary(
        data, numerical_cols, categorical_cols, time_series_cols, text_cols)

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data_prompt = f"""
    Analyze the dataset from the provided CSV file. Below is the summary information:

    Dataset Summary:
    Columns: {', '.join(data.columns)}
    Column Information (Data Types): {json.dumps(column_info, indent=2)}
    Summary Statistics (Key Insights):
    {dataset_summary}

    Based on this, please:
    1. Identify key trends and patterns.
    2. Point out any potential outliers or anomalies.
    3. Suggest any potential insights or analyses that might be valuable.
    4. Provide any other interesting observations from the dataset.

    Kindly return your findings in a MARKDOWN format, you can use html in it to further beautify it,
    highlighting the trends, insights, and any notable outliers or anomalies.
    """

    data_to_send = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "text"},
        "messages": [
            {
                "role": "system",
                "content": """You are an intelligent and experienced data analyst capable of providing insights from datasets.
                    You provide your insights in the form of a story which is very captivating.
                    You always support your claims with data. You only claim when you have data to back it up.
                    You are very professional and well versed in providing insights in comprehensive fashion.
                    You always return your findings in a MARKDOWN format you can use html in it to further beautify it.
                    You will never add placeholder images or dummy images"""
            },
            {
                "role": "user",
                "content": data_prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data_to_send)

    if response.status_code == 200:
        result = response.json()
        analysis = result["choices"][0]["message"]["content"]
        print("Done Analysis")
        return analysis
    else:
        print(response.text)
        print(f"Error: {response.status_code}")
        raise Exception("Didn't work")


@retry(stop=stop_after_attempt(7))
def analyze_and_generate_graphs(data: pd.DataFrame, api_key: str) -> None:
    """
    Analyzes the dataset, identifies data types, and dynamically calls functions to generate graphs
    via function calling using an LLM.

    """

    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(
        include=['object', 'category']).columns.tolist()
    time_series_cols = [
        col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    text_cols = [
        col for col in categorical_cols if data[col].str.len().mean() > 50]

    # Descriptive Statistics for Numerical Columns
    dataset_summary = data_summary(
        data, numerical_cols, categorical_cols, time_series_cols, text_cols)

    functions = [
        {
            "name": "generate_numerical_charts",
            "description": "Generate histograms, box plots, and correlation heatmaps for numerical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerical_cols": {
                        "type": "array",
                        "description": "List of numerical column names.",
                        "items": {"type": "string"}
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "Folder path to save the generated charts."
                    }
                },
                "required": ["numerical_cols", "output_folder"]
            }
        },
        {
            "name": "generate_categorical_charts",
            "description": "Generate bar plots for categorical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categorical_cols": {
                        "type": "array",
                        "description": "List of categorical column names.",
                        "items": {"type": "string"}
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "Folder path to save the generated charts."
                    }
                },
                "required": ["categorical_cols", "output_folder"]
            }
        },
        {
            "name": "generate_time_series_charts",
            "description": "Generate line plots for time-series data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_series_cols": {
                        "type": "array",
                        "description": "List of time-series column names.",
                        "items": {"type": "string"}
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "Folder path to save the generated charts."
                    }
                },
                "required": ["time_series_cols", "output_folder"]
            }
        },
        {
            "name": "generate_text_charts",
            "description": "Generate word clouds for text data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_cols": {
                        "type": "array",
                        "description": "List of text column names.",
                        "items": {"type": "string"}
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "Folder path to save the generated charts."
                    }
                },
                "required": ["text_cols", "output_folder"]
            }
        },
        {
            "name": "generate_geospatial_charts",
            "description": "Generate scatter plots for geospatial data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat_col": {
                        "type": "string",
                        "description": "latitude column",
                        "items": {"type": "string"}
                    },
                    "lon_col": {
                        "type": "string",
                        "description": "longitude column"
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "Folder path to save the generated charts."
                    }
                },
                "required": ["geospatial_cols", "output_folder"]
            }
        },
        {
            "name": "generate_mixed_data_charts",
            "description": "Generate plots for both categorical and numerical data",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_folder": {
                        "type": "string",
                        "description": "Folder path to save the generated charts."
                    }
                },
                "required": ["output_folder"]
            }
        }
    ]

    # API request payload
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    data_prompt = f"""
    Your task is to analyze the dataset and call the appropriate function(s) to generate graphs. Ensure each function
    receives only the relevant columns and parameters required for its operation. Call the most unique function that
    can be used for the given dataset.

    If there are only numerical and categorical columns, call the mixed function
    The dataset summary is {dataset_summary}
    The folder path to be used is {folder_path}.

    """

    data_to_send = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "text"},
        "messages": [
            {
                "role": "system",
                "content": """You are an intelligent data analyst capable of calling functions to analyze
                              datasets and generate relevant graphs. Your job is of atmost priority and any mistake can
                              cause huge losses. So you call functions very intelligently"""
            },
            {
                "role": "user",
                "content": data_prompt
            }
        ],
        "functions": functions,
        "function_call": "auto"
    }

    # Make the API request
    response = requests.post(url, headers=headers, json=data_to_send)

    if response.status_code == 200:
        result = response.json()
        # Extracting the function call result
        function_call_result = result.get("choices", [{}])[0].get(
            "message", {}).get("function_call", {})
        try:
            func = eval(function_call_result.get("name"))
            args = eval(function_call_result.get("arguments"))
            func(data, **args)
        except:
            generate_mixed_data_charts(data, folder_path)
        print("Analysis Done")
        return None
    else:
        print(response.text)
        print(f"Error: {response.status_code}")
        raise Exception("Didn't work")


def data_summary(data, numerical_cols, categorical_cols, time_series_cols, text_cols):
    numerical_summary = data[numerical_cols].describe().T
    numerical_summary['skewness'] = data[numerical_cols].skew()
    numerical_summary['kurtosis'] = data[numerical_cols].apply(kurtosis)

    # Correlation Analysis
    correlation_matrix = data[numerical_cols].corr()

    # Categorical Summary
    categorical_summary = {}
    for col in categorical_cols:
        freq_table = data[col].value_counts(normalize=True).head(5)
        categorical_summary[col] = freq_table.to_dict()

    # Chi-Squared Test for Independence (Example for Pairs of Categorical Variables)
    chi_squared_results = {}
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i + 1:]:
            contingency_table = pd.crosstab(data[col1], data[col2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            chi_squared_results[f"{col1} vs {col2}"] = {
                'chi2': chi2, 'p_value': p}

    # Time-Series Summary (Feature Extraction)
    time_series_summary = {}
    for col in time_series_cols:
        time_series_summary[col] = {
            'start_date': data[col].min(),
            'end_date': data[col].max(),
            'unique_dates': data[col].nunique()
        }

    # Text Summary
    text_summary = {}
    for col in text_cols:
        text_summary[col] = {
            'avg_length': data[col].str.len().mean(),
            'max_length': data[col].str.len().max(),
            'top_words': pd.Series(' '.join(data[col]).split()).value_counts().head(5).to_dict()
        }

    # Final Summary
    dataset_summary = {
        "numerical_summary": numerical_summary.to_dict(),
        "correlation_matrix": correlation_matrix.to_dict(),
        "categorical_summary": categorical_summary,
        "chi_squared_results": chi_squared_results,
        "time_series_summary": time_series_summary,
        "text_summary": text_summary,
    }

    return dataset_summary


def save_readme(file_path: str, llm_response: str) -> None:
    """
    Saves the LLM response and analysis summary in a README.md file inside a folder
    named after the dataset (folder name will be the same as the dataset filename).

    :param file_path: Path to the dataset CSV file.
    :param llm_response: The response from the LLM to be written into the README.
    """
    readme_path = get_readme(file_path)

    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n## LLM Insights\n")
        f.write(f"{llm_response}\n")

    print(f"Analysis complete. Results saved to {readme_path} and charts.")


def get_readme(file_path: str) -> str:
    """
    Generates the file path for the README.md based on the dataset name.

    :param file_path: Path to the dataset file.
    :return: Path to the README.md file inside a folder named after the dataset.
    """
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    readme_path = os.path.join(folder_path, "README.md")
    return readme_path


"""
Chart Functions for Data Analysis
"""

# 1. Numerical Data Charts


def generate_numerical_charts(data: pd.DataFrame, numerical_cols: list, output_folder: str) -> None:
    """
    Generates histograms, box plots, and correlation heatmaps for numerical data.

    """
    for col in numerical_cols:
        # Histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True, bins=30, color='blue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_folder, f"{col}_histogram.png"))
        plt.close()

        # Box Plot
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=data[col], color='green')
        plt.title(f"Box Plot of {col}")
        plt.ylabel(col)
        plt.savefig(os.path.join(output_folder, f"{col}_boxplot.png"))
        plt.close()

    # Correlation Heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = data[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(output_folder, "correlation_matrix.png"))
        plt.close()


# 2. Categorical Data Charts
def generate_categorical_charts(data: pd.DataFrame, categorical_cols: list, output_folder: str) -> None:
    """
    Generates bar plots for categorical data.

    """
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=col, palette="viridis",
                      hue=col, legend=False)
        plt.title(f"Count Plot of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_folder, f"{col}_countplot.png"))
        plt.close()


# 3. Time-Series Data Charts
def generate_time_series_charts(data: pd.DataFrame, time_col: str, value_cols: list, output_folder: str) -> None:
    """
    Generates line plots for time-series data.

    """
    for col in value_cols:
        plt.figure(figsize=(12, 6))
        plt.plot(data[time_col], data[col],
                 marker='o', linestyle='-', label=col)
        plt.title(f"Time-Series Plot of {col}")
        plt.xlabel(time_col)
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{col}_timeseries.png"))
        plt.close()


# 4. Text Data Charts
def generate_text_charts(data: pd.DataFrame, text_cols: str, output_folder: str) -> None:
    """
    Generates a word cloud for text data.

    """
    text_data = " ".join(data[text_cols].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {text_cols}")
    plt.savefig(os.path.join(output_folder, f"{text_cols}_wordcloud.png"))
    plt.close()


# 5. Geospatial Data Charts
def generate_geospatial_charts(data: pd.DataFrame, lat_col: str, lon_col: str, output_folder: str) -> None:
    """
    Generates a scatter plot for geospatial data.

    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[lon_col], data[lat_col], c='red', alpha=0.5)
    plt.title("Geospatial Scatter Plot")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "geospatial_scatterplot.png"))
    plt.close()


# 6. Mixed Data Charts
def generate_mixed_data_charts(data: pd.DataFrame, output_folder: str) -> None:
    """
    Detects data types and generates appropriate charts for each type.

    :param data: Pandas DataFrame containing the dataset.
    :param output_folder: Path to save the generated charts.
    """
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(
        include=['object', 'category']).columns.tolist()

    if numerical_cols:
        generate_numerical_charts(data, numerical_cols, output_folder)
    if categorical_cols:
        generate_categorical_charts(data, categorical_cols, output_folder)


@retry(stop=stop_after_attempt(3))
def analyze_image_with_llm(image_path: str, api_key: str) -> Optional[str]:
    """
    Sends image data with reduced quality to an LLM via the proxy API and returns analysis results in markdown format.

    :param image_path: Path to the image file.
    :param api_key: OpenAI API key for authenticating the request.
    :return: string with the LLM's analysis results, or None if the request fails.
    """
    # Load and compress the image
    try:
        img = Image.open(image_path)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
    except Exception as e:
        print(f"Error loading or compressing image: {e}")
        return None

    # Prepare the API request
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data_prompt = f"""
    Analyze the provided correlation matrix and provide some insights. Heading should be correlation matrix analysis

    Return your findings in a MARKDOWN format, using HTML to enhance the readability of the results.
    """

    # Convert image bytes to a base64 string
    import base64
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    data_to_send = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "text"},
        "messages": [
            {
                "role": "system",
                "content": """You are an intelligent image analyst capable of providing insights from images.
                    You describe the image in detail and provide interesting observations.
                    You always return your findings in a MARKDOWN format, using HTML to enhance readability.
                    Avoid placeholder or dummy content."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": data_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
    }

    # Send the request
    response = requests.post(url, headers=headers, json=data_to_send)

    if response.status_code == 200:
        result = response.json()
        analysis = result["choices"][0]["message"]["content"]
        return analysis
    else:
        print(response.text)
        print(f"Error: {response.status_code}")
        raise Exception("Didn't work")


def main(file_path: str) -> None:
    """
    Initiates the loading, summarizing, and analyzing of the dataset.
    It generates summary statistics, visualizations, and interacts with an LLM for further analysis.

    """
    data = load_data(file_path)

    numerical_summary, categorical_summary, missing_values = summarize_data(
        data)

    readme_path = get_readme(file_path)

    with open(readme_path, "w") as f:
        f.write("# Data Summary\n")
        f.write("## Numerical Summary\n")
        f.write(numerical_summary.to_markdown())
        f.write("\n## Categorical Summary\n")
        f.write(categorical_summary.to_markdown())
        f.write("\n## Missing Values\n")
        f.write(missing_values.to_markdown())

    corr_path = generate_visualizations(data, file_path)
    llm_response = None
    try:
        llm_response = analyze_image_with_llm(corr_path, AIPROXY_TOKEN)
    except:
        pass
    if llm_response:
        save_readme(file_path, llm_response)
    try:
        llm_response = analyze_with_llm(file_path, AIPROXY_TOKEN)
    except:
        pass
    if llm_response:
        save_readme(file_path, llm_response)
    try:
        analyze_and_generate_graphs(data, AIPROXY_TOKEN)
    except:
        pass
    print("Analysis complete. Results saved to README.md and charts.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a dataset filename.")
    else:
        file_path = sys.argv[1]
        main(file_path)
