import configargparse
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from src.server.server import Server

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


def main():
    # fmt:off
    parser = configargparse.ArgParser(default_config_files=["config.yml"])
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    parser.add_argument('--converted_model_name_or_path', type=str, default='bigscience/test-bloomd-6b3',
                        help="path or name of a pretrained model, converted with cli/convert_model.py (see README.md)")
    parser.add_argument('--num_blocks', type=int, default=None, help="The number of blocks to serve")
    parser.add_argument('--block_indices', type=str, default=None, help="Specific block indices to serve")
    parser.add_argument('--prefix', type=str, default=None, help="Announce all blocks with this prefix. By default,"
                                                                 "use the same name as in the converted model.")
    parser.add_argument('--host_maddrs', nargs='+', default=['/ip4/0.0.0.0/tcp/0'], required=False,
                        help='Multiaddrs to listen for external connections from other p2p instances; default: all IPv4 and TCP: /ip4/0.0.0.0/tcp/0')
    parser.add_argument('--announce_maddrs', nargs='+', default=None, required=False,
                        help='Visible multiaddrs the host announces for external connections from other p2p instances')

    parser.add_argument('--compression', type=str, default='NONE', required=False, help='Tensor compression communication')

    parser.add_argument('--num_handlers', type=int, default=16, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--min_batch_size', type=int, default=1,
                        help='Minimum required batch size for all expert operations')
    parser.add_argument('--max_batch_size', type=int, default=16384,
                        help='The total number of examples in the same batch will not exceed this value')
    parser.add_argument('--cache_size_bytes', type=int, default=None,
                        help='The size of memory cache for storing past attention keys/values between inference steps')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all experts will use this device in torch notation; default: cuda if available else cpu')
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        help="Use this dtype to store block weights and do computations. "
                             "By default, respect the dtypes in the pre-trained state dict.")

    parser.add_argument('--throughput',
                        type=lambda value: value if value in ['auto', 'eval'] else float(value),
                        default='auto',
                        help='Expected server throughput (a float measured in RPS). '
                             'If set to "auto" (default), the script evaluates network and compute throughput '
                             'on the first run and uses these estimates for future runs. '
                             'If set to "eval", the script re-evaluates the throughput and overrides the cache.')
    parser.add_argument('--update_period', type=float, required=False, default=30,
                        help='Server will report experts to DHT once in this many seconds')
    parser.add_argument('--expiration', type=float, required=False, default=None,
                        help='DHT entries will expire after this many seconds')
    parser.add_argument('--initial_peers', type=str, nargs='*', required=False, default=[],
                        help='multiaddrs of one or more active DHT peers (if you want to join an existing DHT)')
    parser.add_argument('--increase_file_limit', action='store_true',
                        help='On *nix, this will increase the max number of processes '
                             'a server can spawn before hitting "Too many open files"; Use at your own risk.')
    parser.add_argument('--stats_report_interval', type=int, required=False,
                        help='Interval between two reports of batch processing performance statistics')

    parser.add_argument('--custom_module_path', type=str, required=False,
                        help='Path of a file with custom nn.modules, wrapped into special decorator')
    parser.add_argument('--identity_path', type=str, required=False, help='Path to identity file to be used in P2P')
    parser.add_argument("--use_auth_token", type=str, default=None, help="auth token for from_pretrained")

    # fmt:on
    args = vars(parser.parse_args())
    args.pop("config", None)

    if args.pop("increase_file_limit"):
        increase_file_limit()

    compression_type = args.pop("compression")
    compression = getattr(CompressionType, compression_type)

    use_auth_token = args.pop("use_auth_token")
    args["use_auth_token"] = True if use_auth_token in ("True", "true", "") else use_auth_token

    server = Server(**args, compression=compression, start=True)

    try:
        server.join()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
