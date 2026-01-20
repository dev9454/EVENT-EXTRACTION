# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:05:08 2024

@author: mfixlz
"""

import pandas as pd
import sys
import numpy as np
from asammdf.blocks.utils import (master_using_raster, UniqueDB,
                                  TERMINATED, components, downcast)
from functools import reduce


def setup_fs():
    """Helper function to setup the file system for the examples.
    """
    from fsspec.implementations.local import LocalFileSystem

    fs = LocalFileSystem()

    return fs


def _to_dataframe(
    self,
    channels=None,
    raster=None,
    time_from_zero: bool = True,
    empty_channels="skip",
    keep_arrays: bool = False,
    use_display_names: bool = False,
    time_as_date: bool = False,
    reduce_memory_usage: bool = False,
    raw: bool = False,
    ignore_value2text_conversions: bool = False,
    use_interpolation: bool = False,
    only_basenames: bool = False,
    interpolate_outwards_with_nan: bool = False,
    numeric_1D_only: bool = False,
    progress=None,
) -> pd.DataFrame:
    """generate pandas DataFrame

    Parameters
    ----------
    channels : list
        list of items to be filtered (default None); each item can be :

            * a channel name string
            * (channel name, group index, channel index) list or tuple
            * (channel name, group index) list or tuple
            * (None, group index, channel index) list or tuple

    raster : float | np.array | str
        new raster that can be

        * a float step value
        * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
        * an array (starting with asammdf 5.5.0)

        see `resample` for examples of using this argument

    time_from_zero : bool
        adjust time channel to start from 0; default *True*
    empty_channels : str
        behaviour for channels without samples; the options are *skip* or
        *zeros*; default is *skip*
    use_display_names : bool
        use display name instead of standard channel name, if available.
    keep_arrays : bool
        keep arrays and structure channels as well as the
        component channels. If *True* this can be very slow. If *False*
        only the component channels are saved, and their names will be
        prefixed with the parent channel.
    time_as_date : bool
        the dataframe index will contain the datetime timestamps
        according to the measurement start time; default *False*. If
        *True* then the argument ``time_from_zero`` will be ignored.
    reduce_memory_usage : bool
        reduce memory usage by converting all float columns to float32 and
        searching for minimum dtype that can reprezent the values found
        in integer columns; default *False*
    raw (False) : bool
        the columns will contain the raw values

        .. versionadded:: 5.7.0

    ignore_value2text_conversions (False) : bool
        valid only for the channels that have value to text conversions and
        if *raw=False*. If this is True then the raw numeric values will be
        used, and the conversion will not be applied.

        .. versionadded:: 5.8.0

    use_interpolation (True) : bool
        option to perform interpoaltions when multiple timestamp raster are
        present. If *False* then dataframe columns will be automatically
        filled with NaN's were the dataframe index values are not found in
        the current column's timestamps

        .. versionadded:: 5.11.0

    only_basenames (False) : bool
        use just the field names, without prefix, for structures and channel
        arrays

        .. versionadded:: 5.13.0

    interpolate_outwards_with_nan : bool
        use NaN values for the samples that lie outside of the original
        signal's timestamps

        .. versionadded:: 5.15.0

    Returns
    -------
    dataframe : pandas.DataFrame

    """
    # self.__class__.to_dataframe(self)
    use_interpolation = False
    # print('^^^^^^^^^^^^^^^^^^^^^^ direct inside')
    if channels is not None:
        mdf = self.filter(channels)

        result = mdf.to_dataframe(
            raster=raster,
            time_from_zero=time_from_zero,
            empty_channels=empty_channels,
            keep_arrays=keep_arrays,
            use_display_names=use_display_names,
            time_as_date=time_as_date,
            reduce_memory_usage=reduce_memory_usage,
            raw=raw,
            ignore_value2text_conversions=ignore_value2text_conversions,
            use_interpolation=use_interpolation,
            only_basenames=only_basenames,
            interpolate_outwards_with_nan=interpolate_outwards_with_nan,
            numeric_1D_only=numeric_1D_only,
        )

        mdf.close()
        return result

    target_byte_order = "<=" if sys.byteorder == "little" else ">="

    df = {}

    self._set_temporary_master(None)

    if raster is not None:
        try:
            raster = float(raster)
            assert raster > 0
        except (TypeError, ValueError):
            if isinstance(raster, str):
                raster = self.get(raster).timestamps
            else:
                raster = np.array(raster)
        else:
            raster = master_using_raster(self, raster)
        master = raster
    else:
        masters = {index: self.get_master(index)
                   for index in self.virtual_groups}

        if masters:
            master = reduce(np.union1d, masters.values())
        else:
            master = np.array([], dtype="<f4")

        del masters

    idx = np.argwhere(np.diff(master, prepend=-np.inf) > 0).flatten()
    master = master[idx]

    used_names = UniqueDB()
    used_names.get_unique_name("timestamps")

    groups_nr = len(self.virtual_groups)

    if progress is not None:
        if callable(progress):
            progress(0, groups_nr)
        else:
            progress.signals.setValue.emit(0)
            progress.signals.setMaximum.emit(groups_nr)

            if progress.stop:
                return TERMINATED

    for group_index, (virtual_group_index, virtual_group) in enumerate(self.virtual_groups.items()):
        if virtual_group.cycles_nr == 0 and empty_channels == "skip":
            continue

        channels = [
            (None, gp_index, ch_index)
            for gp_index, channel_indexes in self.included_channels(virtual_group_index)[
                virtual_group_index
            ].items()
            for ch_index in channel_indexes
            if ch_index != self.masters_db.get(gp_index, None)
        ]

        signals = self.select(channels, raw=True,
                              copy_master=False, validate=False)

        if not signals:
            continue

        group_master = signals[0].timestamps

        for sig in signals:
            if len(sig) == 0:
                if empty_channels == "zeros":
                    sig.samples = np.zeros(
                        len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                        dtype=sig.samples.dtype,
                    )
                    sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

        if not raw:
            for signal in signals:
                conversion = signal.conversion
                if conversion:
                    samples = conversion.convert(
                        signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                    )
                    signal.samples = samples

                signal.raw = False
                signal.conversion = None
                if signal.samples.dtype.kind == "S":
                    signal.encoding = "utf-8" if self.version >= "4.00" else "latin-1"

        for s_index, sig in enumerate(signals):
            sig = sig.validate(copy=False)

            if len(sig) == 0:
                if empty_channels == "zeros":
                    sig.samples = np.zeros(
                        len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                        dtype=sig.samples.dtype,
                    )
                    sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

            signals[s_index] = sig

        if use_interpolation:
            same_master = np.array_equal(master, group_master)

            if not same_master and interpolate_outwards_with_nan:
                idx = np.argwhere((master >= group_master[0]) & (
                    master <= group_master[-1])).flatten()

            cycles = len(group_master)

            signals = [
                (
                    signal.interp(
                        master,
                        integer_interpolation_mode=self._integer_interpolation,
                        float_interpolation_mode=self._float_interpolation,
                    )
                    if not same_master or len(signal) != cycles
                    else signal
                )
                for signal in signals
            ]

            if not same_master and interpolate_outwards_with_nan:
                for sig in signals:
                    sig.timestamps = sig.timestamps[idx]
                    sig.samples = sig.samples[idx]

            group_master = master

        if any(len(sig) for sig in signals):
            signals = [sig for sig in signals if len(sig)]

        if group_master.dtype.byteorder not in target_byte_order:
            group_master = group_master.byteswap().view(group_master.dtype.newbyteorder())

        if signals:
            diffs = np.diff(group_master, prepend=-np.inf) > 0
            if np.all(diffs):
                index = pd.Index(group_master, tupleize_cols=False)

            else:
                idx = np.argwhere(diffs).flatten()
                group_master = group_master[idx]

                index = pd.Index(group_master, tupleize_cols=False)

                for sig in signals:
                    sig.samples = sig.samples[idx]
                    sig.timestamps = sig.timestamps[idx]
        else:
            index = pd.Index(group_master, tupleize_cols=False)

        size = len(index)
        for sig in signals:
            if sig.timestamps.dtype.byteorder not in target_byte_order:
                sig.timestamps = sig.timestamps.byteswap().view(
                    sig.timestamps.dtype.newbyteorder())

            sig_index = index if len(sig) == size else pd.Index(
                sig.timestamps, tupleize_cols=False)

            # byte arrays
            if len(sig.samples.shape) > 1:
                if use_display_names:
                    channel_name = list(sig.display_names)[
                        0] if sig.display_names else sig.name
                else:
                    channel_name = sig.name

                channel_name = used_names.get_unique_name(channel_name)

                if sig.samples.dtype.byteorder not in target_byte_order:
                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                df[channel_name] = pd.Series(
                    list(sig.samples),
                    index=sig_index,
                )

            # arrays and structures
            elif sig.samples.dtype.names:
                for name, series in components(
                    sig.samples,
                    sig.name,
                    used_names,
                    master=sig_index,
                    only_basenames=only_basenames,
                ):
                    df[name] = series

            # scalars
            else:
                if use_display_names:
                    channel_name = list(sig.display_names)[
                        0] if sig.display_names else sig.name
                else:
                    channel_name = sig.name

                channel_name = used_names.get_unique_name(channel_name)

                if reduce_memory_usage and sig.samples.dtype.kind not in "SU":
                    if sig.samples.size > 0:
                        sig.samples = downcast(sig.samples)

                if sig.samples.dtype.byteorder not in target_byte_order:
                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                df[channel_name] = pd.Series(
                    sig.samples, index=sig_index, fastpath=True)

        if progress is not None:
            if callable(progress):
                progress(group_index + 1, groups_nr)
            else:
                progress.signals.setValue.emit(group_index + 1)

                if progress.stop:
                    return TERMINATED

    strings, nonstrings = {}, {}

    for col, series in df.items():
        if series.dtype.kind == "S":
            strings[col] = series
        else:
            nonstrings[col] = series

    if numeric_1D_only:
        nonstrings = {col: series for col,
                      series in nonstrings.items() if series.dtype.kind in "uif"}
        strings = {}
    try:
        df = pd.DataFrame(nonstrings, index=master)
    except:
        print('default method of ASAMDF is failing using monkey patching')
        nonstrings = {key: series.rename(key)
                      for key, series in nonstrings.items()}
        if bool(nonstrings):
            df = pd.concat(list(nonstrings.values()), axis=1)
        else:
            df = pd.DataFrame(nonstrings, index=master)

    if strings:
        df_strings = pd.DataFrame(strings, index=master)
        df = pd.concat([df, df_strings], axis=1)

    df.index.name = "timestamps"

    if time_as_date:
        delta = pd.to_timedelta(df.index, unit="s")

        new_index = self.header.start_time + delta
        df.set_index(new_index, inplace=True)

    elif time_from_zero and len(master):
        df.set_index(df.index - df.index[0], inplace=True)

    return df


def _to_dataframe2(
    self,
    channels=None,
    raster=None,
    time_from_zero: bool = True,
    empty_channels="skip",
    keep_arrays: bool = False,
    use_display_names: bool = False,
    time_as_date: bool = False,
    reduce_memory_usage: bool = False,
    raw: bool = False,
    ignore_value2text_conversions: bool = False,
    use_interpolation: bool = True,
    only_basenames: bool = False,
    interpolate_outwards_with_nan: bool = False,
    numeric_1D_only: bool = False,
    progress=None,
) -> pd.DataFrame:
    """generate pandas DataFrame

    Parameters
    ----------
    channels : list
        list of items to be filtered (default None); each item can be :

            * a channel name string
            * (channel name, group index, channel index) list or tuple
            * (channel name, group index) list or tuple
            * (None, group index, channel index) list or tuple

    raster : float | np.array | str
        new raster that can be

        * a float step value
        * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
        * an array (starting with asammdf 5.5.0)

        see `resample` for examples of using this argument

    time_from_zero : bool
        adjust time channel to start from 0; default *True*
    empty_channels : str
        behaviour for channels without samples; the options are *skip* or
        *zeros*; default is *skip*
    use_display_names : bool
        use display name instead of standard channel name, if available.
    keep_arrays : bool
        keep arrays and structure channels as well as the
        component channels. If *True* this can be very slow. If *False*
        only the component channels are saved, and their names will be
        prefixed with the parent channel.
    time_as_date : bool
        the dataframe index will contain the datetime timestamps
        according to the measurement start time; default *False*. If
        *True* then the argument ``time_from_zero`` will be ignored.
    reduce_memory_usage : bool
        reduce memory usage by converting all float columns to float32 and
        searching for minimum dtype that can reprezent the values found
        in integer columns; default *False*
    raw (False) : bool
        the columns will contain the raw values

        .. versionadded:: 5.7.0

    ignore_value2text_conversions (False) : bool
        valid only for the channels that have value to text conversions and
        if *raw=False*. If this is True then the raw numeric values will be
        used, and the conversion will not be applied.

        .. versionadded:: 5.8.0

    use_interpolation (True) : bool
        option to perform interpoaltions when multiple timestamp raster are
        present. If *False* then dataframe columns will be automatically
        filled with NaN's were the dataframe index values are not found in
        the current column's timestamps

        .. versionadded:: 5.11.0

    only_basenames (False) : bool
        use just the field names, without prefix, for structures and channel
        arrays

        .. versionadded:: 5.13.0

    interpolate_outwards_with_nan : bool
        use NaN values for the samples that lie outside of the original
        signal's timestamps

        .. versionadded:: 5.15.0

    Returns
    -------
    dataframe : pandas.DataFrame

    """
    # self.__class__.to_dataframe(self)
    # use_interpolation = False
    # print('^^^^^^^^^^^^^^^^^^^^^^ direct inside')
    if channels is not None:
        mdf = self.filter(channels)

        result = mdf.to_dataframe(
            raster=raster,
            time_from_zero=time_from_zero,
            empty_channels=empty_channels,
            keep_arrays=keep_arrays,
            use_display_names=use_display_names,
            time_as_date=time_as_date,
            reduce_memory_usage=reduce_memory_usage,
            raw=raw,
            ignore_value2text_conversions=ignore_value2text_conversions,
            use_interpolation=use_interpolation,
            only_basenames=only_basenames,
            interpolate_outwards_with_nan=interpolate_outwards_with_nan,
            numeric_1D_only=numeric_1D_only,
        )

        mdf.close()
        return result

    target_byte_order = "<=" if sys.byteorder == "little" else ">="

    df = {}

    self._set_temporary_master(None)

    if raster is not None:
        try:
            raster = float(raster)
            assert raster > 0
        except (TypeError, ValueError):
            if isinstance(raster, str):
                raster = self.get(raster).timestamps
            else:
                raster = np.array(raster)
        else:
            raster = master_using_raster(self, raster)
        master = raster
    else:
        masters = {index: self.get_master(index)
                   for index in self.virtual_groups}

        if masters:
            master = reduce(np.union1d, masters.values())
        else:
            master = np.array([], dtype="<f4")

        del masters

    idx = np.argwhere(np.diff(master, prepend=-np.inf) > 0).flatten()
    master = master[idx]

    used_names = UniqueDB()
    used_names.get_unique_name("timestamps")

    groups_nr = len(self.virtual_groups)

    if progress is not None:
        if callable(progress):
            progress(0, groups_nr)
        else:
            progress.signals.setValue.emit(0)
            progress.signals.setMaximum.emit(groups_nr)

            if progress.stop:
                return TERMINATED

    for group_index, (virtual_group_index, virtual_group) in enumerate(self.virtual_groups.items()):
        if virtual_group.cycles_nr == 0 and empty_channels == "skip":
            continue

        channels = [
            (None, gp_index, ch_index)
            for gp_index, channel_indexes in self.included_channels(virtual_group_index)[
                virtual_group_index
            ].items()
            for ch_index in channel_indexes
            if ch_index != self.masters_db.get(gp_index, None)
        ]

        signals = self.select(channels, raw=True,
                              copy_master=False, validate=False)

        if not signals:
            continue

        group_master = signals[0].timestamps

        for sig in signals:
            if len(sig) == 0:
                if empty_channels == "zeros":
                    sig.samples = np.zeros(
                        len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                        dtype=sig.samples.dtype,
                    )
                    sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

        if not raw:
            for signal in signals:
                conversion = signal.conversion
                if conversion:
                    samples = conversion.convert(
                        signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                    )
                    signal.samples = samples

                signal.raw = False
                signal.conversion = None
                if signal.samples.dtype.kind == "S":
                    signal.encoding = "utf-8" if self.version >= "4.00" else "latin-1"

        for s_index, sig in enumerate(signals):
            sig = sig.validate(copy=False)

            if len(sig) == 0:
                if empty_channels == "zeros":
                    sig.samples = np.zeros(
                        len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                        dtype=sig.samples.dtype,
                    )
                    sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

            signals[s_index] = sig

        if use_interpolation:
            same_master = np.array_equal(master, group_master)

            if not same_master and interpolate_outwards_with_nan:
                idx = np.argwhere((master >= group_master[0]) & (
                    master <= group_master[-1])).flatten()

            cycles = len(group_master)

            signals = [
                (
                    signal.interp(
                        master,
                        integer_interpolation_mode=self._integer_interpolation,
                        float_interpolation_mode=self._float_interpolation,
                    )
                    if not same_master or len(signal) != cycles
                    else signal
                )
                for signal in signals
            ]

            if not same_master and interpolate_outwards_with_nan:
                for sig in signals:
                    sig.timestamps = sig.timestamps[idx]
                    sig.samples = sig.samples[idx]

            group_master = master

        if any(len(sig) for sig in signals):
            signals = [sig for sig in signals if len(sig)]

        if group_master.dtype.byteorder not in target_byte_order:
            group_master = group_master.byteswap().view(group_master.dtype.newbyteorder())

        if signals:
            diffs = np.diff(group_master, prepend=-np.inf) > 0
            if np.all(diffs):
                index = pd.Index(group_master, tupleize_cols=False)

            else:
                idx = np.argwhere(diffs).flatten()
                group_master = group_master[idx]

                index = pd.Index(group_master, tupleize_cols=False)

                for sig in signals:
                    sig.samples = sig.samples[idx]
                    sig.timestamps = sig.timestamps[idx]
        else:
            index = pd.Index(group_master, tupleize_cols=False)

        size = len(index)
        for sig in signals:
            if sig.timestamps.dtype.byteorder not in target_byte_order:
                sig.timestamps = sig.timestamps.byteswap().view(
                    sig.timestamps.dtype.newbyteorder())

            sig_index = index if len(sig) == size else pd.Index(
                sig.timestamps, tupleize_cols=False)

            # byte arrays
            if len(sig.samples.shape) > 1:
                if use_display_names:
                    channel_name = list(sig.display_names)[
                        0] if sig.display_names else sig.name
                else:
                    channel_name = sig.name

                channel_name = used_names.get_unique_name(channel_name)

                if sig.samples.dtype.byteorder not in target_byte_order:
                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                df[channel_name] = pd.Series(
                    list(sig.samples),
                    index=sig_index,
                )

            # arrays and structures
            elif sig.samples.dtype.names:
                for name, series in components(
                    sig.samples,
                    sig.name,
                    used_names,
                    master=sig_index,
                    only_basenames=only_basenames,
                ):
                    df[name] = series

            # scalars
            else:
                if use_display_names:
                    channel_name = list(sig.display_names)[
                        0] if sig.display_names else sig.name
                else:
                    channel_name = sig.name

                channel_name = used_names.get_unique_name(channel_name)

                if reduce_memory_usage and sig.samples.dtype.kind not in "SU":
                    if sig.samples.size > 0:
                        sig.samples = downcast(sig.samples)

                if sig.samples.dtype.byteorder not in target_byte_order:
                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                df[channel_name] = pd.Series(
                    sig.samples, index=sig_index, fastpath=True)

        if progress is not None:
            if callable(progress):
                progress(group_index + 1, groups_nr)
            else:
                progress.signals.setValue.emit(group_index + 1)

                if progress.stop:
                    return TERMINATED

    strings, nonstrings = {}, {}

    for col, series in df.items():
        if series.dtype.kind == "S":
            strings[col] = series
        else:
            nonstrings[col] = series

    if numeric_1D_only:
        nonstrings = {col: series for col,
                      series in nonstrings.items() if series.dtype.kind in "uif"}
        strings = {}
    try:
        df = pd.DataFrame(nonstrings, index=master)
    except:
        print('default method of ASAMDF is failing using monkey patching')
        nonstrings = {key: series.rename(key)
                      for key, series in nonstrings.items()}
        if bool(nonstrings):
            df = pd.concat(list(nonstrings.values()), axis=1)
        else:
            df = pd.DataFrame(nonstrings, index=master)

    if strings:
        df_strings = pd.DataFrame(strings, index=master)
        df = pd.concat([df, df_strings], axis=1)

    df.index.name = "timestamps"

    if time_as_date:
        delta = pd.to_timedelta(df.index, unit="s")

        new_index = self.header.start_time + delta
        df.set_index(new_index, inplace=True)

    elif time_from_zero and len(master):
        df.set_index(df.index - df.index[0], inplace=True)

    return df
