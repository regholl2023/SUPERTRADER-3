
from deep_translator import GoogleTranslator
langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)
print(langs_dict)