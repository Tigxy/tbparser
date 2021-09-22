import os.path
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Union, Optional

import imageio
import numpy as np

from tbparser.events_reader import EventReadingError, EventsFileReader

SummaryItem = namedtuple(
    'SummaryItem', ['run', 'tag', 'step', 'wall_time', 'value', 'type']
)

ScalarItem = namedtuple(
    "ScalarItem", ['step', 'wall_time', 'value']
)


def _get_scalar(value) -> Optional[np.ndarray]:
    """
    Decode an scalar event
    :param value: A value field of an event
    :return: Decoded scalar
    """
    if value.HasField('simple_value'):
        return value.simple_value
    return None


def _get_image(value) -> Optional[np.ndarray]:
    """
    Decode an image event
    :param value: A value field of an event
    :return: Decoded image
    """
    if value.HasField('image'):
        encoded_image = value.image.encoded_image_string
        data = imageio.imread(encoded_image)
        return data
    return None


def _get_image_raw(value) -> Optional[np.ndarray]:
    """
    Return raw image data
    :param value: A value field of an event
    :return: Raw image data
    """
    if value.HasField('image'):
        return value.image.encoded_image_string
    return None


class SummaryReader(Iterable):
    """
    Iterates over events in all the files in the current logdir.
    Only scalars and images are supported at the moment.
    """

    _DECODERS = {
        'scalar': _get_scalar,
        'image': _get_image,
        'image_raw': _get_image_raw,
    }

    def __init__(
            self,
            logdir: Union[str, Path],
            tag_filter: Optional[Iterable] = None,
            types: Iterable = ('scalar',),
            stop_on_error: bool = False
    ):
        """
        Initalize new summary reader
        :param logdir: A directory with Tensorboard summary data
        :param tag_filter: A list of tags to leave (`None` for all)
        :param types: A list of types to get.
            Only 'scalar' and 'image' types are allowed at the moment.
        :param stop_on_error: Whether stop on a broken file
        """
        self._logdir = Path(logdir)

        self._tag_filter = set(tag_filter) if tag_filter is not None else None
        self._types = set(types)
        self._check_type_names()
        self._stop_on_error = stop_on_error

    def _check_type_names(self):
        if self._types is None:
            return
        if not all(
                type_name in self._DECODERS.keys() for type_name in self._types
        ):
            raise ValueError('Invalid type name')

    def _decode_events(self, events: Iterable, run=None) -> Optional[SummaryItem]:
        """
        Convert events to `SummaryItem` instances
        :param events: An iterable with events objects
        :return: A generator with decoded events
            or `None`s if an event can't be decoded
        """

        for event in events:
            if not event.HasField('summary'):
                yield None
            step = event.step
            wall_time = event.wall_time
            for value in event.summary.value:
                tag = value.tag
                for value_type in self._types:
                    decoder = self._DECODERS[value_type]
                    data = decoder(value)
                    if data is not None:
                        yield SummaryItem(
                            run=run,
                            tag=tag,
                            step=step,
                            wall_time=wall_time,
                            value=data,
                            type=value_type
                        )
                else:
                    yield None

    def _check_tag(self, tag: str) -> bool:
        """
        Check if a tag matches the current tag filter
        :param tag: A string with tag
        :return: A boolean value.
        """
        return self._tag_filter is None or tag in self._tag_filter

    def _check_item(self, item):
        return

    def load_scalar_data(self):
        tag_data = defaultdict(lambda: defaultdict(lambda: list()))
        for item in self:
            tag_data[item.tag][item.run].append(ScalarItem(
                step=item.step,
                wall_time=item.wall_time,
                value=item.value
            ))

        # ensure that values are sorted by time
        for tag_key in tag_data.keys():
            for run_key in tag_data[tag_key].keys():
                tag_data[tag_key][run_key].sort(key=lambda x: x.wall_time)
                
        for tag_key in tag_data.keys():
            tag_data[tag_key] = dict(tag_data[tag_key])
        
        return dict(tag_data)

    def __iter__(self) -> SummaryItem:
        """
        Iterate over events in all the files in the current logdir
        :return: A generator with `SummaryItem` objects
        """
        log_files = sorted(f for f in self._logdir.glob(os.path.join('**', '*')) if f.is_file())
        for file_path in log_files:
            with open(file_path, 'rb') as f:
                reader = EventsFileReader(f)
                try:
                    run_name = os.path.relpath(file_path, self._logdir)
                    run_name = os.path.dirname(run_name)
                    yield from (
                        item for item in self._decode_events(reader, run_name)
                        if item is not None and all([
                            self._check_tag(item.tag),
                            item.type in self._types
                        ])
                    )
                except EventReadingError:
                    if self._stop_on_error:
                        raise
                    else:
                        continue
