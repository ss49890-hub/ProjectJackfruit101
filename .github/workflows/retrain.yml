name: Auto Retrain

on:
  schedule:
    - cron: '0 0 * * 0'  # ทุกอาทิตย์ วันอาทิตย์
  workflow_dispatch:  # กด trigger เองได้ด้วย

jobs:
  retrain:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install ffmpeg
        run: sudo apt-get install -y ffmpeg

      - name: Install dependencies
        run: pip install supabase==1.2.0 librosa tensorflow-cpu==2.13.0 scikit-learn numpy soundfile

      - name: Download data from Supabase
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
        run: python download_data.py

      - name: Retrain model
        run: python retrain.py

      - name: Commit new model
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add model_cnn4.tflite
          git commit -m "Auto retrain model" || echo "No changes"
          git push
