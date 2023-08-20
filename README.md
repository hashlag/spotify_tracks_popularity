# Spotify track popularity prediction with linear regression

This model tries to predict the popularity score of a track on Spotify based on
[Spotify tracks DB dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)

## Run model

You should have `pandas`, `torch` and  `sklearn` installed.

Then you can navigate to the project directory and run the model with:
```
python linear.py
```

## Testing

Here is a visualization of MAE change during training

![train loss plot](https://raw.githubusercontent.com/hashlag/spotify_tracks_popularity/main/train_loss_plot.png)

To get actual MAE on testing data just run the `linear.py` script.