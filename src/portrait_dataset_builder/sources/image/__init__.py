"""Image source providers — re-export base types and auto-register plugins."""

from portrait_dataset_builder.sources.image.base import ImageResult, ImageSource
from portrait_dataset_builder.sources.image.bing import BingImageSource
from portrait_dataset_builder.sources.image.duckduckgo import DuckDuckGoImageSource
from portrait_dataset_builder.sources.image.flickr import FlickrImageSource
from portrait_dataset_builder.sources.image.google import GoogleImageSource
from portrait_dataset_builder.sources.image.imdb import IMDbImageSource
from portrait_dataset_builder.sources.image.pexels import PexelsImageSource
from portrait_dataset_builder.sources.image.pixabay import PixabayImageSource
from portrait_dataset_builder.sources.image.unsplash import UnsplashImageSource
from portrait_dataset_builder.sources.image.wikimedia import WikimediaImageSource
from portrait_dataset_builder.sources.image.wikipedia import WikipediaImageSource

__all__ = [
    "ImageResult",
    "ImageSource",
    "GoogleImageSource",
    "BingImageSource",
    "DuckDuckGoImageSource",
    "FlickrImageSource",
    "UnsplashImageSource",
    "PixabayImageSource",
    "PexelsImageSource",
    "WikimediaImageSource",
    "WikipediaImageSource",
    "IMDbImageSource",
]
