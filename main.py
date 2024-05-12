
from exploration.description import *
from analysis import *
import string

def text_reivew():
    # input your data with relative path
    customer_feedback = pd.read_excel('data/CUSTOMER_FEEDBACK.xlsx', sheet_name='Sheet')

    # Data  Visualisatoin
    customer_feedback = EDA(customer_feedback)

    # Date time adjust
    customer_feedback = date_process(customer_feedback,'SURVEY_TIME')


    visulaisation(customer_feedback)
    return customer_feedback

def clean_chinese_review(text):
  """
  This function cleans a Chinese customer review text.

  Args:
      text: The Chinese text review to be cleaned.

  Returns:
      The cleaned Chinese text review.
  """
  # Segment the text into words
  seg_list = jieba.cut(text)
  # Remove stop words (optional)
  stopwords = set("的 地 是 在 一 个 我 你 了 等 等 啊 吗 于 以 对 其 不 上 面 下 要 就 因为 因为 而 不过 虽然 然而 但是 然后 就 例如 例如 话 比如 说 这样 或者")  # Example stop words list (modify as needed)
  filtered_words = [word for word in seg_list if word not in stopwords]
  # Remove punctuation and symbols
  cleaned_text = ''.join([char for char in ''.join(filtered_words) if char not in string.punctuation])
  # Lowercase the text (optional)
  # cleaned_text = cleaned_text.lower()  # Uncomment for lowercase conversion
  return cleaned_text




if __name__ == '__main__':
    # input your data with relative path
    customer_feedback = pd.read_excel('data/CUSTOMER_FEEDBACK.xlsx', sheet_name='Sheet')

    # Data  Visualisatoin
    customer_feedback = EDA(customer_feedback)

    # Date time adjust
    customer_feedback = date_process(customer_feedback,'SURVEY_TIME')
    print(customer_feedback.head(10))

    # visulaisation(customer_feedback)
    # Example usage
    review = "这部手机太棒了！屏幕清晰，拍照功能强大，而且价格实惠！强烈推荐！"
    cleaned_review = clean_chinese_review(review)
    print(f"Original review: {review}")
    print(f"Cleaned review: {cleaned_review}")






