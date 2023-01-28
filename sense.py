#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
from collections import defaultdict
from csv import DictWriter
from datetime import datetime, timezone
from io import StringIO
from json import dumps
from time import time, sleep, monotonic


class Sensors:
    MAX_ULL = 2 ** 64

    def __init__(self):
        self._io_count = 0
        self._io_cache = {}

    def _cached_io(self, io, *paths):
        cache = self._io_cache
        now = monotonic()
        value = None
        error = None
        v = cache.get(paths)
        if v is None or (now - v[0]) > 60:
            try:
                value = io(os.path.join(*paths))
            except BaseException as e:
                error = e
            cache[paths] = now, value, error
        else:
            _, value, error = v

        self._io_count += 1
        if self._io_count >= 300:
            for k, (t, v, e) in tuple(cache.items()):
                if now - t > 3600:
                    cache.pop(k)
            self._io_count = 0

        if error is None:
            return value
        raise error

    def _list_dir(self, *paths):
        return tuple(sorted(os.listdir(os.path.join(*paths))))

    #
    # The idiom:
    #
    #   with open(...) as f:
    #     return f.read()
    #
    # Produces plenty of syscalls that in our use case are really unnecessary:
    #
    #   openat(AT_FDCWD, "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", O_RDONLY|O_CLOEXEC) = 3
    #   fstat(3, {st_mode=S_IFREG|0444, st_size=4096, ...}) = 0
    #   ioctl(3, TCGETS, 0x7fd9221130)          = -1 ENOTTY (Inappropriate ioctl for device)
    #   lseek(3, 0, SEEK_CUR)                   = 0
    #   ioctl(3, TCGETS, 0x7fd9220f30)          = -1 ENOTTY (Inappropriate ioctl for device)
    #   lseek(3, 0, SEEK_CUR)
    #   fstat(3, {st_mode=S_IFREG|0444, st_size=4096, ...}) = 0
    #   read(3, "600000\n", 4097)               = 7
    #   read(3, "", 4090)
    #   close(3)
    #
    # The _read_string below produces:
    #
    #   openat(AT_FDCWD, "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", O_RDONLY|O_CLOEXEC) = 3
    #   read(3, "600000\n", 16384)
    #   close(3)
    #
    # We use a 16kB buffer, for large buffers Python tries to optimize i/o with memory mapping, which produces:
    #
    #   openat(AT_FDCWD, "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", O_RDONLY|O_CLOEXEC) = 3
    #   mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f8b400000
    #   read(3, "600000\n", 1048576)            = 7
    #   mremap(0x7f8b400000, 1052672, 4096, MREMAP_MAYMOVE) = 0x7f8b400000
    #   close(3)
    #
    def _read_string(self, *paths, default=None):
        fd = None
        chunk_size = 16 * 1024
        try:
            fd = os.open(os.path.join(*paths), os.O_RDONLY)
            content = b''
            while True:
                chunk = os.read(fd, chunk_size)
                content += chunk
                if len(chunk) < chunk_size:
                    break
            return content.decode().strip()
        except IOError:
            pass
        finally:
            if fd is not None:
                os.close(fd)
        return default

    def _read_integer(self, *paths):
        try:
            return int(self._read_string(*paths))
        except TypeError:
            pass

    def read_cpu_ticks(self, path='/proc/stat'):
        for i in self._read_string(path, default='').splitlines():
            if not i.startswith('cpu'):
                continue

            tokens = i.strip().split()
            cpu_name = tokens[0]
            if cpu_name == 'cpu':
                continue

            cpu_number = int(cpu_name[3:]) + 1
            cpu_stats = tuple(int(i) for i in tokens[1:])
            cpu_total_ticks = sum(cpu_stats)
            # Consider idle+iowait ticks as spare cpu cycles and everything else as used cpu cycles
            # See https://github.com/torvalds/linux/blob/master/fs/proc/stat.c
            cpu_spare_ticks = cpu_stats[3] + cpu_stats[4]
            yield cpu_number, cpu_total_ticks, cpu_spare_ticks

    def read_cpu_freqs(self, root='/sys/devices/system/cpu'):
        try:
            for cpu_name in self._cached_io(self._list_dir, root):
                if not cpu_name.startswith('cpu'):
                    continue

                cpu_number = cpu_name[3:]
                if cpu_number.isdecimal():
                    cpu_freq = self._read_integer(root, cpu_name, 'cpufreq', 'scaling_cur_freq')
                    if cpu_freq is not None:
                        yield int(cpu_number) + 1, cpu_freq * 1000
        except OSError:
            pass

    def read_power_caps(self, root='/sys/class/powercap'):
        try:
            for powercap_name in self._cached_io(self._list_dir, root):
                powercap_description = self._cached_io(self._read_string, root, powercap_name, 'name')
                if not powercap_description:
                    continue
                powercap_value = self._read_integer(root, powercap_name, 'energy_uj')
                if powercap_value is not None:
                    powercap_max = self._cached_io(self._read_integer, root, powercap_name, 'max_energy_range_uj')
                    yield powercap_name, powercap_description, powercap_value, powercap_max
        except OSError:
            pass

    def read_hardware_sensors(self, root='/sys/class/hwmon', sensors_re=re.compile(r'^((temp|fan)(\d+))_input$')):
        try:
            for i in self._cached_io(self._list_dir, root):
                if not i.startswith('hwmon'):
                    continue
                hwmon_number = i[5:]
                if not hwmon_number.isdecimal():
                    continue
                hwmon_number = int(hwmon_number)

                inputs = set()

                files = set(self._cached_io(self._list_dir, root, i))
                for j in files:
                    match = sensors_re.match(j)
                    if match:
                        inputs.add(match.groups())

                sensor_name = self._cached_io(self._read_string, root, i, 'name')
                if not sensor_name:
                    continue

                input_offset = 0
                if sensor_name == 'coretemp':
                    sensor_device = os.readlink(os.path.join(root, i, 'device'))
                    _, hwmon_number = sensor_device.rsplit('.', maxsplit=1)
                    hwmon_number = int(hwmon_number) + 1
                    sensor_name = 'core'
                    input_offset = -1

                for input_name, input_type, input_number in sorted(inputs):
                    input_number = int(input_number) + input_offset
                    input_file = input_name + '_input'
                    input_value = self._read_integer(root, i, input_file)
                    if input_value is None:
                        continue
                    description_file = input_name + '_label'
                    input_description = None
                    if description_file in files:
                        input_description = self._cached_io(self._read_string, root, i, description_file)
                    offset_file = input_name + '_offset'
                    if offset_file in files:
                        offset_value = self._read_integer(root, i, offset_file)
                        if offset_value is not None:
                            input_value += offset_value
                    if input_type == 'temp':
                        input_value = float(input_value) / 1000 + 273.15
                        yield sensor_name, hwmon_number, input_number, input_description, input_value, 'temp', 'K'
                    elif input_type == 'fan':
                        yield sensor_name, hwmon_number, input_number, input_description, input_value, 'freq', 'RPM'
        except OSError:
            pass

    def read(self, *, cpu_usage=True, cpu_freq=True, power_caps=True, hardware_sensors=True):
        prev_timestamp = None
        prev_cpu_ticks = None
        prev_power_caps = None

        while True:
            curr_timestamp = monotonic()
            curr_results = []

            if cpu_usage:
                curr_cpu_ticks = {i[0]: i[1:] for i in self.read_cpu_ticks()}
                if prev_cpu_ticks is not None:
                    for instance, curr_v in curr_cpu_ticks.items():
                        prev_v = prev_cpu_ticks.get(instance)
                        if prev_v is not None:
                            curr_total_ticks, curr_spare_ticks = curr_v
                            prev_total_ticks, prev_spare_ticks = prev_v
                            total_ticks = curr_total_ticks - prev_total_ticks
                            while total_ticks < 0:
                                total_ticks += self.MAX_ULL
                            spare_ticks = curr_spare_ticks - prev_spare_ticks
                            while spare_ticks < 0:
                                spare_ticks += self.MAX_ULL
                            curr_results.append({
                                'resource': 'cpu',
                                'instance': (instance,),
                                'description': None,
                                'quantity': 'usage',
                                'value': 100 - 100 * spare_ticks / total_ticks,
                                'unit': '%',
                            })
                prev_cpu_ticks = curr_cpu_ticks

            if power_caps:
                curr_power_caps = {i[0]: i[1:] for i in self.read_power_caps()}
                if prev_power_caps is not None:
                    dt = curr_timestamp - prev_timestamp
                    for instance, curr_v in curr_power_caps.items():
                        prev_v = prev_power_caps.get(instance)
                        if prev_v is not None:
                            curr_description, curr_energy, curr_max = curr_v
                            prev_description, prev_energy, prev_max = prev_v
                            energy_used = curr_energy - prev_energy
                            while energy_used < 0:
                                energy_used += 1 + curr_max
                            curr_results.append({
                                'resource': 'power',
                                'instance': (instance,),
                                'description': curr_description,
                                'quantity': 'power',
                                'value': energy_used / dt / 1000000.0,
                                'unit': 'W',
                            })
                prev_power_caps = curr_power_caps

            if cpu_freq:
                for instance, freq in self.read_cpu_freqs():
                    curr_results.append({
                        'resource': 'cpu',
                        'instance': (instance,),
                        'description': None,
                        'quantity': 'freq',
                        'value': freq,
                        'unit': 'Hz',
                    })

            if hardware_sensors:
                for name, hwmon, instance, description, value, quantity, unit in self.read_hardware_sensors():
                    curr_results.append({
                        'resource': name,
                        'instance': (hwmon, instance),
                        'description': description,
                        'quantity': quantity,
                        'value': value,
                        'unit': unit,
                    })

            prev_timestamp = curr_timestamp
            yield curr_results


def _convert_temperature_to_k(datapoint):
    if datapoint['quantity'] == 'temp' and datapoint['unit'] == 'K':
        return datapoint


def _convert_temperature_to_c(datapoint):
    if datapoint['quantity'] == 'temp' and datapoint['unit'] == 'K':
        datapoint['value'] -= 273.15
        datapoint['unit'] = 'C'
        return datapoint


def _convert_temperature_to_f(datapoint):
    if datapoint['quantity'] == 'temp' and datapoint['unit'] == 'K':
        datapoint['value'] = datapoint['value'] * 1.8 - 459.67
        datapoint['unit'] = 'F'
        return datapoint


def _convert_frequency_to_mhz(datapoint):
    if datapoint['quantity'] == 'freq' and datapoint['unit'] == 'Hz':
        datapoint['value'] /= 1000000
        datapoint['unit'] = 'MHz'
        return datapoint


def _convert_frequency_to_ghz(datapoint):
    if datapoint['quantity'] == 'freq' and datapoint['unit'] == 'Hz':
        datapoint['value'] /= 1000000000
        datapoint['unit'] = 'GHz'
        return datapoint


def _convert_power_to_kw(datapoint):
    if datapoint['quantity'] == 'power' and datapoint['unit'] == 'W':
        datapoint['value'] /= 1000
        datapoint['unit'] = 'kW'
        return datapoint


def _convert_power_to_mw(datapoint):
    if datapoint['quantity'] == 'power' and datapoint['unit'] == 'W':
        datapoint['value'] *= 1000
        datapoint['unit'] = 'mW'
        return datapoint


def _format_ndjson(timestamp, datapoints):
    timestamp = round(timestamp * 1000)
    for i in datapoints:
        i['timestamp'] = timestamp
    return dumps(datapoints)


def _format_csv(timestamp, datapoints):
    timestamp = round(timestamp * 1000)
    columns = ('timestamp', 'resource', 'instance', 'quantity', 'value', 'unit', 'description')
    with StringIO() as buffer:
        csv = DictWriter(buffer, columns, restval='', extrasaction='ignore')
        csv.writeheader()
        for i in datapoints:
            i['timestamp'] = timestamp
            instance = i['instance']
            if instance:
                i['instance'] = '/'.join(map(str, instance))
            csv.writerow(i)
        return buffer.getvalue()


def _format_simple(timestamp, datapoints):
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone()
    lines = ['', dt.isoformat()]
    resources = defaultdict(list)
    for datapoint in datapoints:
        resources[(datapoint['resource'], datapoint['instance'])].append(datapoint)
    for k in sorted(resources):
        resource, instance = k
        name = '{}/{}'.format(resource, '/'.join(map(str, instance))) if instance else resource
        cells = []
        description = None
        for i in resources[k]:
            cells.append('{}={}{}'.format(i['quantity'], round(i['value'], 3), i['unit']))
            if description is None:
                description = i.get('description')
        description = ' ({})'.format(description) if description else ''
        lines.append('   {}{} {}'.format(name, description, ' '.join(cells)))
    return '\n'.join(lines)


def _format_table(timestamp, datapoints):
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone()
    lines = ['', dt.isoformat(), '']
    resources = defaultdict(dict)
    column_names = set()
    for datapoint in datapoints:
        resource = datapoint['resource']
        instance = datapoint['instance']
        description = datapoint['description']
        k = resource, instance
        if instance:
            instance = '/'.join(map(str, instance))
        quantity = datapoint['quantity']
        column_names.add(quantity)
        value = datapoint['value']
        unit = datapoint['unit']
        if isinstance(value, int):
            value = '{}{}'.format(value, unit)
        else:
            value = '{:.3f}{}'.format(value, unit)
        datapoint = resources[k]
        datapoint['resource'] = resource
        datapoint['instance'] = instance or ''
        datapoint['description'] = description or ''
        datapoint[quantity] = value
    resources = tuple(resources[k] for k in sorted(resources))
    column_names = sorted(column_names)
    column_names.insert(0, 'instance')
    column_names.insert(0, 'resource')
    column_names.append('description')
    formats = []
    columns = []
    for name in column_names:
        column = [name, '']
        for resource in resources:
            column.append(resource.get(name, ''))
        column_width = max(map(len, column))
        if name == column_names[-1]:
            formats.append('   {{:<{}}}'.format(column_width))
        else:
            formats.append('   {{:>{}}}'.format(column_width))
        columns.append(column)
    line_format = ''.join(formats)
    lines.extend(map(line_format.format, *columns))
    return '\n'.join(lines)


def main(
        interval=10,
        count=0,
        output_format='ndjson',
        converters=(),
        cpu_usage=True,
        cpu_freq=True,
        power_caps=True,
        hardware_sensors=True,
):
    first = True
    sensors = Sensors()
    formatter = globals().get('_format_{}'.format(output_format), _format_simple)
    if count == 0:
        count = -1

    for datapoints in sensors.read(
            cpu_usage=cpu_usage,
            cpu_freq=cpu_freq,
            power_caps=power_caps,
            hardware_sensors=hardware_sensors,
    ):
        if first:
            first = False
            sleep(min(interval, 1))
            last_timestamp = time()
            next_timestamp = last_timestamp + interval
            continue

        for i in datapoints:
            for j in converters:
                if j(i):
                    break

        print(formatter(last_timestamp, datapoints), flush=True)

        if count > 0:
            count -= 1
        if count == 0:
            break

        curr_timestamp = time()
        pause = max(next_timestamp - curr_timestamp, 0.1)
        sleep(pause)
        next_timestamp += interval
        last_timestamp = time()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Power, temperature and usage monitor.')
    parser.add_argument('--interval', metavar='SECONDS', type=float, default='10',
                        help='The interval between updates. Default is 10 seconds.')
    parser.add_argument('--count', metavar='N', type=int, default=0,
                        help='Number of updates. When zero (which is default) repeat forever.')
    parser.add_argument('--output-format', choices=['ndjson', 'csv', 'simple', 'table'], default='simple',
                        help='Choice of the output format. Default is simple.')
    parser.add_argument('--skip-cpu-usage', action='store_true', default=False,
                        help='Skip CPU usage.')
    parser.add_argument('--skip-cpu-freq', action='store_true', default=False,
                        help='Skip CPU frequencies.')
    parser.add_argument('--skip-power-usage', action='store_true', default=False,
                        help='Skip power usage sensors.')
    parser.add_argument('--skip-hardware-sensors', action='store_true', default=False,
                        help='Skip available hardware sensors.')
    parser.add_argument('--freq-unit', choices=['Hz', 'MHz', 'GHz'], default='GHz',
                        help='Choice of frequency units. Default is GHz.')
    parser.add_argument('--temp-unit', choices=['C', 'K', 'F'], default='C',
                        help='Choice of temperature units. Default is C.')
    parser.add_argument('--power-unit', choices=['mW', 'W', 'kW'], default='W',
                        help='Choice of power units. Default is W.')
    args = parser.parse_args()

    converters = []
    if args.freq_unit == 'MHz':
        converters.append(_convert_frequency_to_mhz)
    elif args.freq_unit == 'GHz':
        converters.append(_convert_frequency_to_ghz)
    if args.temp_unit == 'C':
        converters.append(_convert_temperature_to_c)
    elif args.temp_unit == 'F':
        converters.append(_convert_temperature_to_f)
    if args.power_unit == 'mW':
        converters.append(_convert_power_to_mw)
    elif args.freq_unit == 'kW':
        converters.append(_convert_power_to_kw)

    main(
        interval=min(max(0.1, args.interval), 24 * 3600),
        count=args.count,
        output_format=args.output_format,
        converters=converters,
        cpu_usage=not args.skip_cpu_usage,
        cpu_freq=not args.skip_cpu_freq,
        power_caps=not args.skip_power_usage,
        hardware_sensors=not args.skip_hardware_sensors,
    )
