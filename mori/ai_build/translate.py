from deep_translator import GoogleTranslator

def translate_lib(text, to_lang='en'):
    """
    Translates the input text from the source language to the target language.

    :param text: The text to be translated
    :param to_lang: The language code of the target language (default is English 'en')
    :return: The translated text.
    """
    translation = GoogleTranslator(source='auto', target=to_lang).translate(text)
    return translation