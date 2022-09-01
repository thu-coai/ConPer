cd ../src/preprocess
echo 'nltk download corpus...'
python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"
echo 'aggregate...'
python process.py
echo 'split scene...'
python split.py
echo 'get target sentence...'
python keep_one_card.py
echo 'extract keywords...'
python extract_emotion_event.py
