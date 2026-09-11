# Data-Software
"""
Data Quality & Analytics Platform
Part 1: Data Upload Module

This file contains ONLY the Data Upload functionality. Future modules
(Data Quality Report, Analytics/Charts, AI Chat, export, database, etc.)
are intentionally NOT implemented here. See README.md for the roadmap.

Everything in this file is plain Python. Streamlit is used purely as the
UI layer; all logic (validation, parsing, error handling) lives in
ordinary Python functions so it can be reused/tested independently of
the UI later on.
"""

# Import Streamlit to build the web application interface (buttons,
# file uploader, tables, messages, layout, etc.).
import streamlit as st

# Import Pandas, which is the core data structure we load every uploaded
# dataset into. All downstream modules (quality checks, analytics) will
# operate on the DataFrame produced here.
import pandas as pd

# Import io.BytesIO so we can hand uploaded file bytes to libraries
# (pypdf, Pillow) that expect a file-like object rather than raw bytes.
from io import BytesIO

# Import pypdf to extract text from uploaded PDF files.
from pypdf import PdfReader

# Import Pillow's Image module to open uploaded image files and read
# their basic metadata (dimensions, format, color mode).
from PIL import Image


# -----------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------

# The list of file extensions this module currently accepts. Centralizing
# this here means both the uploader widget and the validation function
# stay in sync automatically.
SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls", "pdf", "txt", "jpg", "jpeg"]


# -----------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------

def validate_file(uploaded_file):
    """
    Check that the uploaded file has a supported extension and is not
    empty (zero bytes). This runs BEFORE we attempt any parsing, so we
    can reject obviously bad uploads early with a clear message instead
    of letting a parsing library raise a confusing low-level error.

    Returns a tuple: (is_valid: bool, extension: str, error_message: str)
    """
    # Pull the file name from the Streamlit UploadedFile object and get
    # its extension in lowercase so ".CSV" and ".csv" are both accepted.
    filename = uploaded_file.name
    extension = filename.split(".")[-1].lower() if "." in filename else ""

    # Reject any extension that is not in our supported list. This is
    # what "Prevent unsupported file types from being processed" means
    # in practice.
    if extension not in SUPPORTED_EXTENSIONS:
        return False, extension, (
            f"'.{extension}' is not a supported file type. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS).upper()}."
        )

    # Reject files with no content at all. uploaded_file.size is provided
    # by Streamlit and reflects the number of bytes received.
    if uploaded_file.size == 0:
        return False, extension, "The uploaded file is empty (0 bytes)."

    # If we reach this point, the file passed basic validation.
    return True, extension, ""


# -----------------------------------------------------------------------
# LOADERS - one function per supported file type
# -----------------------------------------------------------------------
# Each loader is responsible ONLY for turning the uploaded bytes into a
# Pandas DataFrame (or raising a descriptive exception). Keeping these
# separate makes it easy to add a new file type later without touching
# unrelated code.

def load_csv(uploaded_file):
    """Read an uploaded CSV file into a DataFrame."""
    # pandas.read_csv understands the Streamlit UploadedFile object
    # directly because it behaves like a file handle.
    df = pd.read_csv(uploaded_file)
    return df


def load_excel(uploaded_file, extension):
    """
    Read an uploaded Excel file (.xlsx or .xls) into a DataFrame.

    Pandas picks the correct parsing engine automatically based on the
    file extension (openpyxl for .xlsx, xlrd for legacy .xls), as long
    as both packages are installed, which requirements.txt guarantees.
    """
    df = pd.read_excel(uploaded_file)
    return df


def load_txt(uploaded_file):
    """
    Read an uploaded .txt file into a DataFrame.

    Plain text files aren't guaranteed to be tabular, so we try a couple
    of reasonable strategies:
      1. Check whether the first line consistently uses one of a small
         set of real delimiter characters (comma, tab, semicolon, pipe)
         across the first several lines. If so, load it as delimited
         data with that specific delimiter.
      2. Otherwise, fall back to loading it as one row per line of text
         under a "line_text" column, so the user still gets something
         useful to inspect.

    Note: we deliberately do NOT use Pandas' `sep=None` auto-sniffing
    here, because it can misfire on ordinary prose (e.g., splitting on
    stray spaces) and produce misleading "columns" out of plain text.
    """
    # Read the raw bytes once so we can try multiple parsing strategies
    # without needing to re-upload the file.
    raw_bytes = uploaded_file.read()
    text = raw_bytes.decode("utf-8", errors="replace")
    sample_lines = [line for line in text.splitlines()[:10] if line.strip()]

    # Strategy 1: only try well-known delimiter characters, and only
    # trust the result if that delimiter appears the SAME number of
    # times on every sampled line (a strong signal of real tabular
    # structure, as opposed to a delimiter character appearing
    # incidentally inside free-form text).
    candidate_delimiters = [",", "\t", ";", "|"]
    detected_delimiter = None

    for delimiter in candidate_delimiters:
        if not sample_lines:
            break
        counts_per_line = [line.count(delimiter) for line in sample_lines]
        if counts_per_line[0] > 0 and len(set(counts_per_line)) == 1:
            detected_delimiter = delimiter
            break

    if detected_delimiter is not None:
        try:
            df = pd.read_csv(BytesIO(raw_bytes), sep=detected_delimiter)
            if df.shape[1] > 1:
                return df
        except Exception:
            # Fall through to the line-by-line fallback below.
            pass

    # Strategy 2 (fallback): treat the file as unstructured text, one
    # row per line. This guarantees .txt files always load successfully.
    lines = text.splitlines()
    df = pd.DataFrame({"line_text": lines})
    return df


def load_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF and load it into a DataFrame with
    one row per page (columns: page_number, extracted_text).

    PDFs are not naturally tabular, so instead of trying to reconstruct
    tables (which is a much larger feature for a future module), Part 1
    focuses on getting the text content into a structured, inspectable
    form.
    """
    reader = PdfReader(BytesIO(uploaded_file.read()))

    # Guard against PDFs with zero pages (e.g., corrupted files that
    # still pass the initial byte-count check).
    if len(reader.pages) == 0:
        raise ValueError("The PDF file contains no pages.")

    page_numbers = []
    page_texts = []

    # Walk through every page and extract whatever text pypdf can find.
    # Some PDFs (scanned images without OCR) will yield empty strings;
    # that is not treated as an error here, just an empty text field.
    for page_index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        page_numbers.append(page_index)
        page_texts.append(extracted.strip())

    df = pd.DataFrame({"page_number": page_numbers, "extracted_text": page_texts})
    return df


def load_image(uploaded_file):
    """
    Load an uploaded image (.jpg/.jpeg) and represent it as a one-row
    DataFrame of metadata (filename, format, dimensions, color mode).

    Images are not tabular data, so "loading it into a DataFrame" here
    means capturing descriptive metadata about the file rather than
    pixel data. The raw image is also displayed directly in the UI (see
    display_image_preview) so the user can visually confirm the upload.
    """
    image_bytes = uploaded_file.read()
    image = Image.open(BytesIO(image_bytes))

    metadata = {
        "filename": [uploaded_file.name],
        "format": [image.format],
        "width_px": [image.width],
        "height_px": [image.height],
        "color_mode": [image.mode],
        "file_size_kb": [round(len(image_bytes) / 1024, 2)],
    }
    df = pd.DataFrame(metadata)

    # Return both the metadata DataFrame and the PIL Image object so the
    # UI layer can render the actual picture as well.
    return df, image


# -----------------------------------------------------------------------
# DISPATCHER
# -----------------------------------------------------------------------

def load_dataset(uploaded_file, extension):
    """
    Route the uploaded file to the correct loader function based on its
    extension. Returns a tuple: (dataframe, image_or_none).

    Centralizing this routing logic in one place means the rest of the
    app doesn't need to know which loader handles which extension.
    """
    if extension == "csv":
        return load_csv(uploaded_file), None
    elif extension in ("xlsx", "xls"):
        return load_excel(uploaded_file, extension), None
    elif extension == "txt":
        return load_txt(uploaded_file), None
    elif extension == "pdf":
        return load_pdf(uploaded_file), None
    elif extension in ("jpg", "jpeg"):
        df, image = load_image(uploaded_file)
        return df, image
    else:
        # This should be unreachable because validate_file() already
        # rejects unsupported extensions, but it's kept as a safety net.
        raise ValueError(f"No loader available for '.{extension}' files.")


# -----------------------------------------------------------------------
# DISPLAY FUNCTIONS - each renders one section of the results UI
# -----------------------------------------------------------------------

def display_dataset_information(df, filename):
    """
    Show the high-level facts about the dataset: its name, row count,
    and column count. This gives the user an immediate sense of scale
    before they scroll into the detailed preview below.
    """
    # st.subheader() renders a section heading; the string itself is
    # the visible heading text in the app.
    st.subheader("Dataset Overview")

    # st.columns() creates a 3-column layout so the three key metrics
    # sit side-by-side instead of stacking vertically.
    col1, col2, col3 = st.columns(3)

    # st.metric() renders a large labeled number - ideal for quick-glance
    # stats like row/column counts. The label strings ("Dataset",
    # "Rows", "Columns") are what appears above each number in the UI.
    col1.metric("Dataset", filename)
    col2.metric("Rows", f"{df.shape[0]:,}")
    col3.metric("Columns", f"{df.shape[1]:,}")


def display_dataset_preview(df):
    """
    Show a preview of the first several rows of the DataFrame so the
    user can visually confirm the data was read correctly.
    """
    st.subheader("Preview")

    # st.dataframe() renders an interactive, scrollable table. We only
    # show the first 50 rows so very large datasets don't slow down or
    # clutter the browser.
    st.dataframe(df.head(50), use_container_width=True)

    # Let the user know if we're only showing a subset, so they aren't
    # confused about why the table doesn't show every row.
    if df.shape[0] > 50:
        st.caption(f"Showing the first 50 of {df.shape[0]:,} rows.")


def display_column_information(df):
    """
    Show each column's name and detected Pandas data type. This is the
    foundation the future Data Quality Report module will build on
    (e.g., flagging unexpected types).
    """
    st.subheader("Column Information")

    # Build a small summary table: one row per column, showing its name
    # and the dtype Pandas inferred while parsing the file.
    column_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Non-Null Count": [df[col].notna().sum() for col in df.columns],
    })

    st.dataframe(column_info, use_container_width=True, hide_index=True)


def display_basic_summary(df):
    """
    Show a basic statistical summary of the dataset using Pandas'
    built-in describe(). This is intentionally lightweight - full data
    quality scoring and analytics come in later modules.
    """
    st.subheader("Basic Dataset Summary")

    # include="all" makes describe() summarize both numeric and
    # non-numeric (text/categorical) columns, since uploaded datasets
    # may be entirely non-numeric (e.g., text lines from a .txt file).
    summary = df.describe(include="all").transpose()

    st.dataframe(summary, use_container_width=True)

    # Also surface how many missing values exist per column, since
    # that's one of the most immediately useful "data quality" facts
    # even before the dedicated Data Quality Report module exists.
    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())

    if total_missing > 0:
        # st.warning() renders an amber/yellow message box, signaling
        # "worth noting" rather than a hard error.
        st.warning(
            f"This dataset contains {total_missing:,} missing value(s) "
            f"across {int((missing_counts > 0).sum())} column(s)."
        )
    else:
        # st.info() renders a neutral blue message box for informational
        # (non-critical) messages.
        st.info("No missing values were detected in this dataset.")


def display_image_preview(image):
    """
    Show the actual uploaded image alongside its metadata table, since
    metadata alone doesn't let the user visually confirm the upload.
    """
    st.subheader("Image Preview")
    # st.image() renders the picture itself in the browser.
    st.image(image, use_container_width=True)


# -----------------------------------------------------------------------
# MAIN APPLICATION FLOW
# -----------------------------------------------------------------------

def main():
    # st.set_page_config() controls browser-tab-level settings. The
    # "page_title" string is what appears in the browser tab, and
    # layout="wide" lets tables use the full screen width.
    st.set_page_config(page_title="Data Quality & Analytics Platform", layout="wide")

    # st.title() renders the large, bold heading at the top of the app.
    # This string is the main visible name of the whole platform.
    st.title("Data Quality & Analytics Platform")

    # st.caption() renders small gray helper text under the title,
    # orienting the user to what this specific page/module does.
    st.caption("Part 1: Data Upload")

    st.header("Upload Your Dataset")

    # st.file_uploader() creates the actual upload control. The first
    # argument is the user-facing label text shown above the button.
    # The `type` argument restricts which file extensions the browser's
    # file picker will allow the user to select in the first place.
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_EXTENSIONS,
    )

    # st.caption() here documents, for the user, exactly which formats
    # are supported - matching what `type=` enforces above.
    st.caption("Supported formats: CSV, XLSX, XLS, PDF, TXT, JPG, JPEG")

    # Only attempt to process a file once the user has actually uploaded
    # one. Streamlit re-runs this whole script on every interaction, so
    # `uploaded_file` is None until a file is selected.
    if uploaded_file is None:
        return

    filename = uploaded_file.name

    # -------------------------------------------------------------
    # STEP 1: Validate the file before attempting to parse it.
    # -------------------------------------------------------------
    is_valid, extension, error_message = validate_file(uploaded_file)

    if not is_valid:
        # st.error() renders a red message box - reserved for problems
        # that stop processing entirely, like an unsupported file type.
        st.error(error_message)
        return

    # -------------------------------------------------------------
    # STEP 2: Attempt to load the file into a DataFrame. Any failure
    # here is caught and shown as a friendly message instead of a raw
    # Python traceback, per the "don't expose technical details"
    # requirement.
    # -------------------------------------------------------------
    try:
        df, image = load_dataset(uploaded_file, extension)
    except (pd.errors.EmptyDataError, ValueError):
        # Raised by Pandas/our loaders when the file has no readable
        # rows/content at all (e.g., a CSV with only headers, or a
        # zero-page PDF).
        st.error(
            "The file was read, but it doesn't appear to contain any "
            "usable data. Please check the file and try again."
        )
        return
    except (pd.errors.ParserError, UnicodeDecodeError):
        # Raised when the file's content doesn't match its extension
        # (e.g., a corrupted or malformed CSV/Excel file).
        st.error(
            "This file appears to be corrupted or not formatted correctly "
            "for a ." + extension + " file. Please verify the file and "
            "try uploading it again."
        )
        return
    except Exception:
        # Catch-all safety net for any other unexpected parsing error
        # (e.g., a damaged Excel workbook, an unreadable PDF/image).
        # We deliberately do not show the raw exception text to the
        # user, keeping the message friendly and non-technical.
        st.error(
            "Something went wrong while reading this file. It may be "
            "corrupted or in an unexpected format. Please try a "
            "different file."
        )
        return

    # -------------------------------------------------------------
    # STEP 3: Guard against datasets that loaded "successfully" but
    # are structurally empty (no rows or no columns) - this is a
    # separate case from a parsing exception.
    # -------------------------------------------------------------
    if df.shape[1] == 0:
        st.error("This file doesn't contain any columns that could be read.")
        return

    if df.shape[0] == 0:
        # st.warning() (rather than st.error()) because a header-only
        # file is technically valid, just empty - the user may still
        # want to see the column structure.
        st.warning("This dataset has no rows - only column headers were found.")
        return

    # -------------------------------------------------------------
    # STEP 4: Success! Clearly communicate the upload worked, then
    # render every results section.
    # -------------------------------------------------------------
    # st.success() renders a green confirmation message box - this is
    # the "clearly communicate successful uploads" requirement.
    st.success(f"'{filename}' was uploaded and read successfully.")

    display_dataset_information(df, filename)

    # Only images have an accompanying picture to render; every other
    # format leaves `image` as None from the dispatcher above.
    if image is not None:
        display_image_preview(image)

    display_dataset_preview(df)
    display_column_information(df)
    display_basic_summary(df)


# Standard Python entry-point guard: ensures main() only runs when this
# file is executed directly by Streamlit (`streamlit run app.py`), not
# if it's ever imported as a module from a future part of the platform.
if __name__ == "__main__":
    main()
