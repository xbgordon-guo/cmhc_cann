[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD MHC]# cd /lgsl_data/dianxin_telechat/mindformers/
[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD mindformers]# cd /workspace/MHC/
[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD MHC]# ll
total 2724
-rwxr-xr-x 1 1003 1003 2775866 Jul 13 09:07 cann-ops-transformer-custom_ascend910_93_linux-aarch64.run
drwxr-xr-x 3 1003 1003      92 Jul  8 09:07 parallel_core
-rwxr-xr-x 1 1003 1003    8469 Jul 13 08:56 telechat4_40B_pretrain_1p.yaml
[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD MHC]# ./cann-ops-transformer-custom_ascend910_93_linux-aarch64.run 
Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
Uncompressing version:1.0  100%  
[ops_custom] [2026-07-14 06:46:31] [INFO] upgrade framework
[ops_custom] [2026-07-14 06:46:31] [INFO] no need to upgrade ops framework files
[ops_custom] [2026-07-14 06:46:31] [INFO] upgrade op proto
es inc lib 
[ops_custom] [2026-07-14 06:46:31] [INFO] replace or merge old ops op_proto files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] copy new ops op_proto files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] upgrade op impl
ai_core 
[ops_custom] [2026-07-14 06:46:31] [INFO] replace or merge old ops op_impl files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] copy new ops op_impl files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] upgrade op api
include lib 
[ops_custom] [2026-07-14 06:46:31] [INFO] replace or merge old ops op_api files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] copy new ops op_api files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] upgrade uninstall
[ops_custom] [2026-07-14 06:46:31] [INFO] copy new uninstall.sh files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] upgrade version.info
[ops_custom] [2026-07-14 06:46:31] [INFO] copy new version.info files ......
[ops_custom] [2026-07-14 06:46:31] [INFO] no need to upgrade custom.proto files
[ops_custom] [2026-07-14 06:46:31] [INFO] using requirements: when custom module install finished or before you run the custom module, execute the command [ export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH} ] to set the environment path
SUCCESS
[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD MHC]# export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD MHC]# cd /lgsl_data/dianxin_telechat/mindformers/
[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD mindformers]# python run_mindformer.py --config /workspace/MHC/telechat4_40B_pretrain_1p.yaml --use_parallel False --device_id 8
/usr/local/python3.11.13/lib/python3.11/site-packages/numpy/core/getlimits.py:549: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
  setattr(self, word, getattr(machar, word).flat[0])
/usr/local/python3.11.13/lib/python3.11/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
  return self._float_to_str(self.smallest_subnormal)
/usr/local/python3.11.13/lib/python3.11/site-packages/numpy/core/getlimits.py:549: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
  setattr(self, word, getattr(machar, word).flat[0])
/usr/local/python3.11.13/lib/python3.11/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
  return self._float_to_str(self.smallest_subnormal)
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:00.610.000 [mindspore/graph/_mark_deprecated.py:59] Module 'mindspore.common.api' is deprecated from version 2.9.0 and will be removed in a future version, use 'mindspore.graph.api' instead.
2026-07-14 06:47:01,717 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/deepseek3/modeling_deepseek_v3.py:31] - WARNING - Import PyNativeDeepseekV3ForCausalLM failed: No module named 'hyper_parallel'.
2026-07-14 06:47:01,757 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/qwen3/modeling_qwen3.py:35] - WARNING - Import PyNativeQwen3ForCausalLM failed: No module named 'hyper_parallel'.
2026-07-14 06:47:04,934 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/tools/register/template.py:84] - WARNING - The input config swap_config is empty.
2026-07-14 06:47:04,935 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/tools/register/template.py:84] - WARNING - The input config metric is empty.
2026-07-14 06:47:04,935 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/tools/register/template.py:708] - WARNING - Some configs in yaml are useless for train: ['profile_output', 'remove_redundancy']
2026-07-14 06:47:04,935 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/run_mindformer.py:71] - INFO - Running MindFormers in GRAPH_MODE.
2026-07-14 06:47:04,935 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/utils/file_utils.py:50] - INFO - set output path to '/lgsl_data/dianxin_telechat/mindformers/output'
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.936.000 [mindspore/context.py:1338] For 'context.set_context', the parameter 'ascend_config' will be deprecated and removed in a future version. Please use the api mindspore.device_context.ascend.op_precision.precision_mode(),
                                                       mindspore.device_context.ascend.op_precision.op_precision_mode(),
                                                       mindspore.device_context.ascend.op_precision.matmul_allow_hf32(),
                                                       mindspore.device_context.ascend.op_precision.conv_allow_hf32(),
                                                       mindspore.device_context.ascend.op_tuning.op_compile() instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.936.000 [mindspore/context.py:907] For 'context.set_context', 'recompute_comm_overlap' parameter is deprecated, and will be removed in the next version, Please use 'recomputation_communication_overlap' instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.936.000 [mindspore/context.py:907] For 'context.set_context', 'matmul_grad_comm_overlap' parameter is deprecated, and will be removed in the next version, Please use 'grad_matmul_communication_overlap' instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.936.000 [mindspore/context.py:907] For 'context.set_context', 'enable_task_opt' parameter is deprecated, and will be removed in the next version, Please use 'enable_communication_fusion' instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.937.000 [mindspore/context.py:907] For 'context.set_context', 'enable_grad_comm_opt' parameter is deprecated, and will be removed in the next version, Please use 'grad_computation_allreduce_overlap' instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.937.000 [mindspore/context.py:907] For 'context.set_context', 'enable_opt_shard_comm_opt' parameter is deprecated, and will be removed in the next version, Please use 'computation_allgather_overlap' instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.937.000 [mindspore/context.py:907] For 'context.set_context', 'recompute_allgather_overlap_fagrad' parameter is deprecated, and will be removed in the next version, Please use 'grad_fa_allgather_overlap' instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.937.000 [mindspore/context.py:907] For 'context.set_context', 'dataset_broadcast_opt_level' parameter is deprecated, and will be removed in the next version, Please use 'dataset_broadcast_opt_level' instead.
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:04.937.000 [mindspore/context.py:907] For 'context.set_context', 'compute_communicate_fusion_level' parameter is deprecated, and will be removed in the next version, Please use 'computation_communication_fusion_level' instead.
2026-07-14 06:47:04,938 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/core/context/build_context.py:714] - WARNING - affinity_cpu_list will be removed in the near future, please use affinity_config instead.
2026-07-14 06:47:04,938 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/core/context/build_context.py:719] - WARNING - custom bind policy affinity_cpu_list must be dict, but got False.
2026-07-14 06:47:04,952 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/tools/register/template.py:84] - WARNING - The input config swap_config is empty.
2026-07-14 06:47:04,952 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/tools/register/template.py:84] - WARNING - The input config monitor_config is empty.
2026-07-14 06:47:04,952 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/tools/register/template.py:708] - WARNING - Some configs in yaml are useless for train: ['auto_tune', 'autotune_per_step', 'eval_dataset', 'eval_dataset_task', 'filepath_prefix', 'processor']
2026-07-14 06:47:04,952 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:1051] - INFO - Load configs in /lgsl_data/dianxin_telechat/mindformers/configs/general/run_general_task.yaml to build trainer.
2026-07-14 06:47:04,952 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:1087] - INFO - ..........Init Config..........
2026-07-14 06:47:04,952 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/core/parallel_config.py:49] - INFO - initial swap_config from dict: {'swap': False, 'layer_swap': None, 'op_swap': None, 'default_prefetch': 1}
2026-07-14 06:47:04,953 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/core/parallel_config.py:56] - INFO - initial recompute_config from dict: {'recompute': True, 'select_recompute': False, 'parallel_optimizer_comm_recompute': True, 'select_comm_recompute': False, 'mp_comm_recompute': True, 'recompute_slice_activation': False, 'select_recompute_exclude': {'self_attention\\.core_attention\\.flash_attention': True}, 'select_comm_recompute_exclude': False}
2026-07-14 06:47:04,953 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/core/parallel_config.py:62] - INFO - initial parallel_config from dict: {'data_parallel': 1, 'model_parallel': 1, 'context_parallel': 1, 'expert_parallel': 1, 'pipeline_stage': 1, 'micro_batch_num': 1, 'seq_split_num': 1, 'use_seq_parallel': True, 'optimizer_shard': None, 'gradient_aggregation_group': 4, 'vocab_emb_dp': True, 'context_parallel_algo': 'colossalai_cp', 'ulysses_degree_in_cp': 1, 'mem_coeff': 0.1}
2026-07-14 06:47:04,953 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/utils/file_utils.py:50] - INFO - set output path to '/lgsl_data/dianxin_telechat/mindformers/output'
2026-07-14 06:47:04,953 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/utils/file_utils.py:65] - INFO - set strategy path to '/lgsl_data/dianxin_telechat/mindformers/output/strategy/ckpt_strategy_rank_0.ckpt'
2026-07-14 06:47:04,954 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/utils/file_utils.py:73] - INFO - set checkpoint save path to `/lgsl_data/dianxin_telechat/mindformers/output/checkpoint`
2026-07-14 06:47:04,954 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:120] - INFO - host_name: YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD, host_ip: 10.127.10.165
2026-07-14 06:47:04,954 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:128] - INFO - Now Running Task is: text_generation, Model is: telechat4_moe
2026-07-14 06:47:04,954 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:161] - WARNING - Input model name is not in the supported list or unspecified.
2026-07-14 06:47:04,954 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:162] - WARNING - See the list of supported task and model name: ['common', 'glm4_9b']
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:163] - WARNING - The default model config: /lgsl_data/dianxin_telechat/mindformers/configs/general/run_general_task.yaml will now be used for the text_generation task 
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:1157] - INFO - ..........Init Model..........
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:333] - INFO - ==========Trainer Init Success!==========
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:430] - WARNING - sink_size will not be able to set in a future release. Modifying sink_size may cause functional issues when resuming training from a checkpoint.
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:1515] - WARNING - The functionality of setting `resume_training` to a weight filename will be deprecated in future versions.
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:1383] - WARNING - The config `check_for_global_norm` and `global_norm_spike_threshold` is deprecated, please use `global_norm_spike_threshold` in `monitor_config.health_checkpoint` instead.
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config load_checkpoint is deprecated. Please use config load_path in `checkpoint_config` instead.
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config balanced_load is deprecated. Please use config load_balanced in `checkpoint_config` instead.
2026-07-14 06:47:04,955 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config resume_training is deprecated. Please use config no_load_optim in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config reshard_worker_num is deprecated. Please use config reshard_worker_num in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config keep_checkpoint_max is deprecated. Please use config save_max in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config save_checkpoint_steps is deprecated. Please use config save_interleaved_steps in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config save_optimizer is deprecated. Please use config no_save_optim in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config remove_redundancy is deprecated. Please use config save_remove_redundancy in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config async_save is deprecated. Please use config async_save in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config directory is deprecated. Please use config save_path in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:796] - WARNING - The config prefix is deprecated. Please use config prefix in `checkpoint_config` instead.
2026-07-14 06:47:04,956 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/trainer.py:1157] - INFO - ..........Init Model..........
2026-07-14 06:47:04,957 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1030] - INFO - Got dataset config: dataset_strategy: [[1, 1], [1, 1], [1, 1], [1, 1]], column_names: ['input_ids', 'labels', 'loss_mask', 'position_ids'], construct_args_key: ['input_ids', 'labels', 'loss_mask', 'position_ids']
2026-07-14 06:47:04,957 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:940] - WARNING - `full_batch` will be deprecated in future interfaces, use `dataset_strategy` instead.
2026-07-14 06:47:04,957 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:329] - WARNING - The Lazy Inline compilation acceleration feature only works with using gradient_accumulation_steps > 1 when not in pipeline parallel mode (pipeline_stage = 1). Current pipeline stage=1 but gradient_accumulation_steps=1, the feature is disabled by default.
2026-07-14 06:47:04,957 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:281] - INFO - The current parallel mode is stand_alone, batch size per card will not be changed: batch_size_per_card = 1
2026-07-14 06:47:04,957 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:285] - INFO - global_batch_size = batch_size_per_card * device_num * gradient_accumulation_steps = 1 = 1 * 1 * 1
2026-07-14 06:47:04,957 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:297] - INFO - parallel_config will be change to default config: [ParallelConfig]
swap:[ParallelConfig]
backward_prefetch:backward_prefetch
layers:layers
_swap:False
_default_prefetch:1
_layer_swap:[]
_op_swap:{'.*\\.flash_attention': [{'backward_prefetch': 1, 'layers': True}]}

_recompute:[ParallelConfig]
_recompute:True
_select_recompute:False
_select_comm_recompute:False
_parallel_optimizer_comm_recompute:True
_mp_comm_recompute:True
_recompute_slice_activation:False
_select_recompute_exclude:{'self_attention\\.core_attention\\.flash_attention': True}
_select_comm_recompute_exclude:False

select_recompute:False
use_seq_parallel:False
context_parallel_algo:ContextParallelAlgo.COLOSSALAI_CP
ulysses_degree_in_cp:1
mem_coeff:0.1
_optimizer_shard:None
_gradient_aggregation_group:4
_embed_dp_mp_config:[ParallelConfig]
_dp_mp_config:[ParallelConfig]
_data_parallel:1
_model_parallel:1
_context_parallel:1
use_seq_parallel:True
select_recompute:False
context_parallel_algo:ContextParallelAlgo.colossalai_cp

_vocab_emb_dp:True
use_seq_parallel:True
select_recompute:False

_pp_config:[ParallelConfig]
_pipeline_stage:1
_micro_batch_num:1
_seq_split_num:1

_moe_config:[ParallelConfig]
_dpmp:[ParallelConfig]
_data_parallel:1
_model_parallel:1
_context_parallel:1
use_seq_parallel:True
select_recompute:False
context_parallel_algo:ContextParallelAlgo.colossalai_cp

_expert_parallel:1
use_seq_parallel:True
select_recompute:False
enable_deredudency:False
npu_nums_per_device:1

.
2026-07-14 06:47:04,958 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1130] - INFO - .........Build Dataset For Train..........
2026-07-14 06:47:04,958 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:445] - INFO - .........Build Dataset From Config..........
2026-07-14 06:47:04,958 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/causal_language_model_dataset.py:309] - INFO - Now Create Causal Language Model Dataset.
2026-07-14 06:47:04,958 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/base_dataset.py:84] - INFO - Now dataset_strategy is [[1, 1], [1, 1], [1, 1], [1, 1]], shard_id: 0, num_shards: 1
2026-07-14 06:47:04,961 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/blended_datasets/indexed_dataset.py:219] - INFO - Load the _IndexReader from /hpfs/huawei/lql/dsa/single-417/single.idx
2026-07-14 06:47:04,963 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/blended_datasets/indexed_dataset.py:240] - INFO - Extract the sequence lengths
2026-07-14 06:47:04,963 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/blended_datasets/indexed_dataset.py:248] - INFO - Extract the sequence pointers
2026-07-14 06:47:04,964 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/blended_datasets/indexed_dataset.py:259] - INFO - Extract the document indices
2026-07-14 06:47:04,964 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/blended_datasets/indexed_dataset.py:290] - INFO - > total number of sequences: 1
2026-07-14 06:47:04,964 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/blended_datasets/indexed_dataset.py:291] - INFO - > total number of documents: 1
2026-07-14 06:47:04,964 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:441] - INFO - Building new index map
2026-07-14 06:47:04,967 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1134] - INFO - Create train dataset finish, dataset size:1
2026-07-14 06:47:04,967 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:222] - INFO - Will be Training epochs:6, sink_size:1
2026-07-14 06:47:04,967 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/utils.py:224] - INFO - Create training dataset finish, dataset size:1
2026-07-14 06:47:04,968 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1249] - INFO - .........Build Net For Train..........
2026-07-14 06:47:04,968 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:479] - INFO - .........Build Network From Config..........
2026-07-14 06:47:04,968 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:147] - WARNING - Found unsupported HuggingFace arguments in TeleChat4MoeConfig:
2026-07-14 06:47:04,968 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:157] - WARNING - +------------------------+---------------------------------------------+
2026-07-14 06:47:04,968 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:158] - WARNING - | Argument               | Status-Info                                 |
2026-07-14 06:47:04,968 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:159] - WARNING - |:-----------------------|:--------------------------------------------|
2026-07-14 06:47:04,969 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:163] - WARNING - | auto_map               | Useless                                     |
2026-07-14 06:47:04,969 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:163] - WARNING - | torch_dtype            | Useless, replace by compute_dtype           |
2026-07-14 06:47:04,969 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:163] - WARNING - | use_cache              | Useless, enable kv_cache by default in MF   |
2026-07-14 06:47:04,969 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:163] - WARNING - | transformers_version   | Useless                                     |
2026-07-14 06:47:04,969 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:163] - WARNING - | ep_size                | Useless                                     |
2026-07-14 06:47:04,969 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:163] - WARNING - | scoring_func           | Useless                                     |
2026-07-14 06:47:04,969 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/models/model_config_utils.py:165] - WARNING - +------------------------+---------------------------------------------+
[WARNING] DEVICE(56203,ffffa4786be0,python):2026-07-14-06:47:05.560.475 [mindspore/ccsrc/plugin/ascend/res_manager/mem_manager/abstract_ascend_memory_pool_support.cc:48] SetMemPoolBlockSize] Memory pool block size 59055800320 is bigger than currently available maximum memory 57982058496, and the actual effective value will be 57982058496
2026-07-14 06:47:05,567 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/transformer_config_utils.py:689] - WARNING - Converted config has the following warnings:

[Unmapped Keys] (1 problem)
  - following keys:{'qk_rope_head_dim', 'qk_nope_head_dim'} not in mapping.Please check if need add mapping rule or add into @ignore_and_delete_parameter
2026-07-14 06:47:05,567 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/transformer_config_utils.py:608] - INFO - Config Conversion summary:
| Source                                                | Target                                                | Trans Func       | Param Usage | Param Origin                                  |
+=======================================================+=======================================================+==================+=============+===============================================+
| data_parallel = 1                                     | data_parallel_size = 1                                | —                | common      | mindformers                                   |
| context_parallel = 1                                  | context_parallel_size = 1                             | —                | common      | megatron-lm                                   |
| model_parallel = 1                                    | tensor_model_parallel_size = 1                        | —                | common      | megatron-lm                                   |
| expert_parallel = 1                                   | expert_model_parallel_size = 1                        | —                | common      | megatron-lm                                   |
| pipeline_stage = 1                                    | pipeline_model_parallel_size = 1                      | —                | common      | megatron-lm                                   |
| micro_batch_num = 1                                   | micro_batch_num = 1                                   | —                | common      | mindformers                                   |
| seq_split_num = 1                                     | seq_split_num = 1                                     | —                | common      | mindformers                                   |
| use_seq_parallel = False                              | sequence_parallel = False                             | —                | common      | megatron-lm                                   |
| gradient_aggregation_group = 4                        | gradient_aggregation_group = 4                        | —                | common      | mindformers                                   |
| vocab_emb_dp = True                                   | vocab_emb_dp = True                                   | —                | common      | mindformers                                   |
| recompute = True                                      | recompute = True                                      | —                | training    | mindformers                                   |
| select_recompute = False                              | select_recompute = False                              | —                | training    | mindformers                                   |
| parallel_optimizer_comm_recompute = True              | parallel_optimizer_comm_recompute = True              | —                | training    | mindformers                                   |
| mp_comm_recompute = True                              | mp_comm_recompute = True                              | —                | training    | mindformers                                   |
| recompute_slice_activation = False                    | recompute_slice_activation = False                    | —                | training    | mindformers                                   |
| select_recompute_exclude = {'self_attention\\.core... | select_recompute_exclude = {'self_attention\\.core... | —                | training    | mindformers                                   |
| select_comm_recompute_exclude = False                 | select_comm_recompute_exclude = False                 | —                | training    | mindformers                                   |
| select_comm_recompute = False                         | select_comm_recompute = False                         | —                | training    | mindformers                                   |
| context_parallel_algo = colossalai_cp                 | cp_comm_type = all_gather                             | get_cp_comm_type | training    | megatron-lm                                   |
| context_parallel_algo = colossalai_cp                 | context_parallel_algo = colossalai_cp                 | —                | training    | mindformers                                   |
| ulysses_degree_in_cp = 1                              | hierarchical_context_parallel_sizes = 1               | —                | common      | megatron-lm                                   |
| ulysses_degree_in_cp = 1                              | ulysses_degree_in_cp = 1                              | —                | common      | mindformers                                   |
| vocab_size = 131072                                   | vocab_size = 131072                                   | —                | common      | ['mindformers', 'huggingface', 'megatron-lm'] |
| max_position_embeddings = 4096                        | max_position_embeddings = 4096                        | —                | common      | ['huggingface', 'megatron-lm']                |
| hidden_size = 3584                                    | hidden_size = 3584                                    | —                | common      | ['huggingface', 'megatron-lm']                |
| intermediate_size = 9216                              | ffn_hidden_size = 9216                                | —                | common      | megatron-lm                                   |
| moe_intermediate_size = 1024                          | moe_ffn_hidden_size = 1024                            | —                | common      | megatron-lm                                   |
| num_hidden_layers = 4                                 | num_layers = 4                                        | —                | common      | megatron-lm                                   |
| num_nextn_predict_layers = 0                          | mtp_num_layers = 0                                    | —                | training    | megatron-lm                                   |
| num_attention_heads = 32                              | num_attention_heads = 32                              | —                | common      | megatron-lm                                   |
| n_shared_experts = 1                                  | shared_expert_num = 1                                 | —                | inference   | mindformers                                   |
| n_routed_experts = 64                                 | num_moe_experts = 64                                  | —                | common      | mindformers                                   |
| routed_scaling_factor = 2.0                           | moe_router_topk_scaling_factor = 2.0                  | —                | common      | megatron-lm                                   |
| kv_lora_rank = 512                                    | kv_lora_rank = 512                                    | —                | —           | —                                             |
| q_lora_rank = 768                                     | q_lora_rank = 768                                     | —                | —           | —                                             |
| v_head_dim = 128                                      | v_head_dim = 128                                      | —                | —           | —                                             |
| topk_method = noaux_tc                                | topk_method = noaux_tc                                | —                | common      | huggingface                                   |
| n_group = None                                        | moe_router_num_groups = None                          | —                | inference   | megatron-lm                                   |
| topk_group = None                                     | moe_router_group_topk = None                          | —                | inference   | megatron-lm                                   |
| num_experts_per_tok = 4                               | moe_router_topk = 4                                   | —                | common      | megatron-lm                                   |
| moe_layer_freq = 1                                    | moe_layer_freq = 1                                    | —                | common      | ['huggingface', 'megatron-lm']                |
| first_k_dense_replace = 2                             | first_k_dense_replace = 2                             | —                | common      | huggingface                                   |
| norm_topk_prob = True                                 | norm_topk_prob = True                                 | —                | common      | huggingface                                   |
| num_key_value_heads = 32                              | num_query_groups = 32                                 | —                | common      | megatron-lm                                   |
| hidden_act = silu                                     | hidden_act = silu                                     | —                | common      | huggingface                                   |
| initializer_range = 0.02                              | init_method_std = 0.02                                | —                | training    | megatron-lm                                   |
| rms_norm_eps = 1e-06                                  | layernorm_epsilon = 1e-06                             | —                | common      | mindformers                                   |
| rope_theta = 10000.0                                  | rotary_base = 10000.0                                 | —                | common      | mindformers                                   |
| rope_scaling = None                                   | rope_scaling = None                                   | —                | training    | huggingface                                   |
| attention_bias = False                                | add_qkv_bias = False                                  | —                | common      | mindformers                                   |
| attention_dropout = 0.0                               | attention_dropout = 0.0                               | —                | common      | mindformers                                   |
| freeze_transformer_layers = False                     | freeze_transformer_layers = False                     | —                | training    | megatron-lm                                   |
| pad_token_id = None                                   | pad_token_id = None                                   | —                | common      | huggingface                                   |
| quantization_config = None                            | quantization_config = None                            | —                | inference   | mindformers                                   |
| pet_config = None                                     | pet_config = None                                     | —                | training    | mindformers                                   |
| tie_word_embeddings = False                           | tie_word_embeddings = False                           | —                | common      | huggingface                                   |
| seq_length = 4096                                     | seq_length = 4096                                     | —                | common      | mindformers                                   |
| compute_dtype = bfloat16                              | compute_dtype = bfloat16                              | —                | common      | mindformers                                   |
| layernorm_compute_dtype = float32                     | layernorm_compute_dtype = float32                     | —                | common      | mindformers                                   |
| rotary_dtype = float32                                | rotary_dtype = float32                                | —                | common      | mindformers                                   |
| hidden_dropout = 0.0                                  | hidden_dropout = 0.0                                  | —                | common      | megatron-lm                                   |
| use_flash_attention = True                            | use_flash_attention = True                            | —                | inference   | mindformers                                   |
| moe_router_score_function = sigmoid                   | moe_router_score_function = sigmoid                   | —                | training    | megatron-lm                                   |
| moe_router_enable_expert_bias = True                  | moe_router_enable_expert_bias = True                  | —                | common      | megatron-lm                                   |
| moe_router_fusion = True                              | moe_router_fusion = True                              | —                | common      | mindformers                                   |
| normalization = RMSNorm                               | normalization = RMSNorm                               | —                | common      | megatron-lm                                   |
| add_bias_linear = False                               | add_bias_linear = False                               | —                | common      | megatron-lm                                   |
| gated_linear_unit = True                              | gated_linear_unit = True                              | —                | common      | megatron-lm                                   |
| pp_interleave_num = 1                                 | virtual_pipeline_model_parallel_size = 1              | —                | common      | megatron-lm                                   |
| multi_latent_attention = True                         | multi_latent_attention = True                         | —                | common      | mindformers                                   |
| mla_qkv_concat = False                                | mla_qkv_concat = False                                | —                | common      | mindformers                                   |
| qk_layernorm = True                                   | qk_layernorm = True                                   | —                | common      | mindformers                                   |
| params_dtype = float32                                | params_dtype = float32                                | —                | common      | megatron-lm                                   |
| softmax_compute_dtype = float32                       | attention_softmax_in_fp32 = True                      | is_float_32      | common      | megatron-lm                                   |
| softmax_compute_dtype = float32                       | softmax_compute_dtype = float32                       | —                | common      | mindformers                                   |
| mtp_loss_factor = 0.0                                 | mtp_loss_scaling_factor = 0.0                         | —                | training    | megatron-lm                                   |
| position_embedding_type = rope                        | position_embedding_type = rope                        | —                | common      | mindformers                                   |
| rotary_scaling_factor = 1.0                           | rotary_scaling_factor = 1.0                           | —                | —           | —                                             |
| beta_fast = 32                                        | beta_fast = 32                                        | —                | —           | —                                             |
| beta_slow = 1                                         | beta_slow = 1                                         | —                | —           | —                                             |
| mscale = 1                                            | mscale = 1                                            | —                | —           | —                                             |
| mscale_all_dim = 1                                    | mscale_all_dim = 1                                    | —                | —           | —                                             |
| experimental_attention_variant = dsa                  | experimental_attention_variant = dsa                  | —                | training    | megatron-lm                                   |
| dsa_indexer_n_heads = 64                              | dsa_indexer_n_heads = 64                              | —                | training    | megatron-lm                                   |
| dsa_indexer_head_dim = 128                            | dsa_indexer_head_dim = 128                            | —                | training    | megatron-lm                                   |
| dsa_indexer_topk = 2048                               | dsa_indexer_topk = 2048                               | —                | training    | megatron-lm                                   |
| dsa_indexer_loss_coeff = 1.0                          | dsa_indexer_loss_coeff = 1.0                          | —                | training    | megatron-lm                                   |
| dsa_indexer_use_sparse_loss = True                    | dsa_indexer_use_sparse_loss = True                    | —                | training    | megatron-lm                                   |
| enable_hyper_connections = True                       | enable_hyper_connections = True                       | —                | —           | —                                             |
| num_residual_streams = 4                              | num_residual_streams = 4                              | —                | —           | —                                             |
| mhc_sinkhorn_iterations = 20                          | mhc_sinkhorn_iterations = 20                          | —                | —           | —                                             |
| mhc_init_gating_factor = 0.01                         | mhc_init_gating_factor = 0.01                         | —                | —           | —                                             |
| enable_fused_triton_sinkhorn = False                  | enable_fused_triton_sinkhorn = False                  | —                | —           | —                                             |
| enable_fused_mhc_ops = True                           | enable_fused_mhc_ops = True                           | —                | —           | —                                             |
| hc_sinkhorn_clamp = 30.0                              | hc_sinkhorn_clamp = 30.0                              | —                | training    | mindformers                                   |
| enable_cmhc = True                                    | enable_cmhc = True                                    | —                | —           | —                                             |
| cmhc_gamma = 0.8                                      | cmhc_gamma = 0.8                                      | —                | —           | —                                             |
| cmhc_gamma_num_layers = 2                             | cmhc_gamma_num_layers = 2                             | —                | —           | —                                             |
| moe_router_dtype = float32                            | moe_router_dtype = float32                            | —                | common      | megatron-lm                                   |
| moe_shared_expert_intermediate_size = 1024            | moe_shared_expert_intermediate_size = 1024            | —                | common      | megatron-lm                                   |
| moe_grouped_gemm = True                               | moe_grouped_gemm = True                               | —                | common      | megatron-lm                                   |
| use_pad_tokens = True                                 | use_pad_tokens = True                                 | —                | common      | mindformers                                   |
| moe_token_dispatcher_type = alltoall                  | moe_token_dispatcher_type = alltoall                  | —                | training    | megatron-lm                                   |
| router_jitter_noise = 0.01                            | router_jitter_noise = 0.01                            | —                | training    | megatron-lm                                   |
| moe_router_load_balancing_type = seq_aux_loss         | moe_router_load_balancing_type = seq_aux_loss         | —                | training    | megatron-lm                                   |
| moe_aux_loss_coeff = 0.0008                           | moe_aux_loss_coeff = 0.0008                           | —                | training    | mindformers                                   |
| moe_router_bias_update_rate = 0.0001                  | moe_router_bias_update_rate = 0.0001                  | —                | training    | megatron-lm                                   |
| moe_router_force_expert_balance = False               | moe_router_force_expert_balance = False               | —                | common      | mindformers                                   |
| disable_lazy_inline = True                            | disable_lazy_inline = True                            | —                | training    | mindformers                                   |
| gradient_accumulation_steps = 1                       | gradient_accumulation_steps = 1                       | —                | training    | mindformers                                   |
| calculate_per_token_loss = None                       | calculate_per_token_loss = None                       | —                | common      | mindformers                                   |
| batch_size = 1                                        | batch_size = 1                                        | —                | training    | mindformers                                   |
+-------------------------------------------------------+-------------------------------------------------------+------------------+-------------+-----------------------------------------------+
2026-07-14 06:47:05,571 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/transformer_config.py:2114] - WARNING - vocab_emb_dp is not supported in MCore, it will be converted to False automatically.
2026-07-14 06:47:05,572 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/config_converter.py:112] - INFO - The final converted MLATransformerConfig is: {
  'batch_size': 1,
  'parallel_config': None,
  'pet_config': None,
  'context_parallel_algo': 'colossalai_cp',
  'is_dynamic': False,
  'compute_dtype': mindspore.bfloat16,
  'layernorm_compute_dtype': mindspore.float32,
  'rotary_dtype': mindspore.float32,
  'rotary_cos_format': 'rotate_half',
  'bias_swiglu_fusion': False,
  'mla_qkv_concat': False,
  'use_contiguous_weight_layout_attention': False,
  'use_interleaved_weight_layout_mlp': True,
  'normalization': 'RMSNorm',
  'fused_norm': True,
  'add_bias_linear': False,
  'add_mlp_fc1_bias_linear': None,
  'add_mlp_fc2_bias_linear': None,
  'gated_linear_unit': True,
  'use_flash_attention': True,
  'rotary_seq_len_interpolation_factor': None,
  'rope_scaling': None,
  'use_rope_scaling': False,
  'input_layout': 'BSND',
  'sparse_mode': 0,
  'use_alibi_mask': False,
  'use_attn_mask_compression': False,
  'use_eod_attn_mask_compression': False,
  'use_attention_mask': True,
  'use_ring_attention': False,
  'fp16_lm_cross_entropy': False,
  'untie_embeddings_and_output_weights': True,
  'hidden_act': 'silu',
  'mask_func_type': 'attn_mask_fill',
  'track_max_attention_logit': False,
  'track_hc_res_stats': False,
  'hc_sinkhorn_clamp': 30.0,
  'comp_comm_parallel': False,
  'comp_comm_parallel_degree': 2,
  'norm_topk_prob': True,
  'use_fused_ops_topkrouter': False,
  'use_shared_expert_gating': False,
  'topk_method': 'noaux_tc',
  'npu_nums_per_device': 8,
  'use_pad_tokens': True,
  'callback_moe_droprate': False,
  'first_k_dense_replace': 2,
  'moe_router_enable_expert_bias': True,
  'moe_router_force_expert_balance': False,
  'moe_router_score_function': 'sigmoid',
  'moe_router_fusion': True,
  'use_eod_reset': False,
  'print_separate_loss': True,
  'vocab_size': 131072,
  'seq_length': 4096,
  'pad_token_id': 0,
  'ignore_token_id': -100,
  'max_position_embeddings': 4096,
  'sandwich_norm': False,
  'tie_word_embeddings': False,
  'block_size': 16,
  'num_blocks': 512,
  'pre_process': True,
  'post_process': True,
  'dispatch_global_max_bs': 0,
  'attn_reduce_scatter': False,
  'attn_allgather': False,
  'attn_allreduce': True,
  'ffn_reduce_scatter': False,
  'ffn_allgather': False,
  'ffn_allreduce': True,
  'use_alltoall': False,
  'use_fused_mla': False,
  'quantization_config': None,
  'disable_lazy_inline': True,
  'coeff': 0.1,
  'data_parallel_size': 1,
  'tensor_model_parallel_size': 1,
  'pipeline_model_parallel_size': 1,
  'virtual_pipeline_model_parallel_size': 1,
  'sequence_parallel': False,
  'context_parallel_size': 1,
  'hierarchical_context_parallel_sizes': 1,
  'expert_model_parallel_size': 1,
  'expert_tensor_parallel_size': None,
  'micro_batch_num': 1,
  'seq_split_num': 1,
  'gradient_aggregation_group': 4,
  'offset': 0,
  'ulysses_degree_in_cp': 1,
  'vocab_emb_dp': False,
  'gradient_accumulation_steps': 1,
  'params_dtype': mindspore.float32,
  'cpu_offloading': False,
  'cpu_offloading_num_layers': None,
  'op_swap': None,
  'default_prefetch': 1,
  'num_layers': 4,
  'mtp_num_layers': 0,
  'mtp_loss_scaling_factor': 0.0,
  'hidden_size': 3584,
  'num_attention_heads': 32,
  'softmax_scale': None,
  'num_query_groups': 32,
  'ffn_hidden_size': 9216,
  'kv_channels': 112,
  'hidden_dropout': 0.0,
  'attention_dropout': 0.0,
  'fp32_residual_connection': False,
  'apply_residual_connection_post_layernorm': False,
  'layernorm_epsilon': 1e-06,
  'add_qkv_bias': False,
  'activation_func': 'gelu',
  'num_moe_experts': 64,
  'rotary_interleaved': False,
  'calculate_per_token_loss': None,
  'multi_latent_attention': True,
  'position_embedding_type': 'rope',
  'nope_layer_interval': None,
  'rotary_base': 10000.0,
  'partial_rotary_factor': 1.0,
  'qk_layernorm': True,
  'experimental_attention_variant': 'dsa',
  'dsa_indexer_n_heads': 64,
  'dsa_indexer_head_dim': 128,
  'dsa_indexer_topk': 2048,
  'dsa_indexer_loss_coeff': 1.0,
  'dsa_indexer_use_sparse_loss': True,
  'enable_hyper_connections': True,
  'num_residual_streams': 4,
  'mhc_sinkhorn_iterations': 20,
  'mhc_init_gating_factor': 0.01,
  'enable_fused_triton_sinkhorn': False,
  'enable_fused_mhc_ops': True,
  'enable_cmhc': True,
  'cmhc_gamma': 0.8,
  'cmhc_gamma_num_layers': 2,
  'init_method': <function init_method_normal.<locals>.init_ at 0xfffddc3f3b00>,
  'output_layer_init_method': <function scaled_init_method_normal.<locals>.init_ at 0xfffddc3f3ce0>,
  'init_method_std': 0.02,
  'param_init_std_rules': None,
  'apply_query_key_layer_scaling': False,
  'attention_softmax_in_fp32': True,
  'softmax_compute_dtype': mindspore.float32,
  'bias_dropout_fusion': False,
  'apply_rope_fusion': False,
  'recompute': True,
  'select_recompute': False,
  'parallel_optimizer_comm_recompute': True,
  'select_comm_recompute': False,
  'mp_comm_recompute': True,
  'recompute_slice_activation': False,
  'select_recompute_exclude': {},
  'select_comm_recompute_exclude': False,
  'moe_shared_expert_intermediate_size': 1024,
  'moe_shared_expert_overlap': False,
  'moe_layer_freq': [0, 0, 1, 1],
  'moe_ffn_hidden_size': 1024,
  'moe_router_load_balancing_type': 'seq_aux_loss',
  'moe_router_topk': 4,
  'moe_router_num_groups': None,
  'moe_router_group_topk': None,
  'moe_router_pre_softmax': False,
  'moe_router_topk_scaling_factor': 2.0,
  'moe_router_dtype': mindspore.float32,
  'moe_router_bias_update_rate': 0.0001,
  'moe_router_bias_update_rate_factor': None,
  'moe_grouped_gemm': True,
  'moe_aux_loss_coeff': 0.0008,
  'moe_z_loss_coeff': None,
  'router_jitter_noise': 0.01,
  'freeze_transformer_layers': False,
  'group_wise_a2a': False,
  'moe_token_dispatcher_type': 'alltoall',
  'moe_expert_capacity_factor': None,
  'moe_token_drop_policy': 'probs',
  'moe_permute_fusion': False,
  'moe_apply_probs_on_input': False,
  'shared_expert_num': 1,
  'print_expert_load': False,
  'enable_expert_relocation': False,
  'expert_relocation_initial_iteration': 20,
  'expert_relocation_freq': 50,
  'cp_comm_type': 'all_gather',
  'window_size': None,
  'window_attn_skip_freq': None,
  'model_architecture': 'decoder_only',
  'num_encoder_layers': None,
  'num_decoder_layers': None,
  'q_lora_rank': 768,
  'kv_lora_rank': 512,
  'qk_head_dim': 128,
  'qk_pos_emb_head_dim': 64,
  'v_head_dim': 128,
  'rope_type': 'yarn',
  'rotary_percent': 1.0,
  'rotary_scaling_factor': 1.0,
  'beta_fast': 32,
  'beta_slow': 1,
  'mscale': 1,
  'mscale_all_dim': 1
}
2026-07-14 06:47:05,572 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/version_control.py:103] - INFO - The Lazy Inline compilation acceleration feature has been called, and the feature is disabled by default.
2026-07-14 06:47:08,133 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:172] - INFO - num_layers per stage: [[4]]
2026-07-14 06:47:08,134 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:173] - INFO - Accumulated num_layers per stage: [[4]]
2026-07-14 06:47:08,134 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:175] - INFO - Pipeline id list with start_stage: [0, 0, 0, 0]
2026-07-14 06:47:08,134 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:176] - INFO - Interleave id list: [0, 0, 0, 0]
2026-07-14 06:47:08,134 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:194] - INFO - Formative layer_recompute: [[4]]
2026-07-14 06:47:08,134 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:196] - INFO - The configuration of select_recompute_exclude and select_comm_recompute_exclude have the highest priority.
2026-07-14 06:47:08,135 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:202] - INFO - Formative select_recompute: {'mlp\\.shared_experts\\.mul': [[0]], 'mlp\\.shared_experts\\.activation_func\\.silu': [[0]], 'mlp\\.mul': [[0]], 'mlp\\.activation_func\\.silu': [[0]]}
2026-07-14 06:47:08,135 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:203] - INFO - Formative select_comm_recompute: {'.*\\.norm': [[0]]}
2026-07-14 06:47:08,135 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:204] - INFO - Formative select_recompute_exclude: {'self_attention\\.core_attention\\.flash_attention': [[4]]}
2026-07-14 06:47:08,135 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:205] - INFO - Formative select_comm_recompute_exclude: {}
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 0
GD===========>[__init__]gamma: 0.8
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:09.400.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:09.595.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:09.785.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:09.973.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 0
GD===========>[__init__]gamma: 0.8
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:10.196.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:10.197.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:10.199.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:10.199.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
2026-07-14 06:47:10,214 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:397] - INFO - Set full recompute at layer 0
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 1
GD===========>[__init__]gamma: 0.8
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.450.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.470.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.480.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.490.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 1
GD===========>[__init__]gamma: 0.8
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.740.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.750.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.760.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:11.770.00 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
2026-07-14 06:47:11,091 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:397] - INFO - Set full recompute at layer 1
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 2
GD===========>[__init__]gamma: 0.8
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.110.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.111.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.112.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.113.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 2
GD===========>[__init__]gamma: 0.8
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.138.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.139.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.140.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:15.141.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
2026-07-14 06:47:15,156 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:397] - INFO - Set full recompute at layer 2
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 3
GD===========>[__init__]gamma: 1.0
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.155.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.156.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.158.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.159.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
GD===========>[__init__]cmhc_gamma_num_layers: 2
GD===========>[__init__]layer_number: 3
GD===========>[__init__]gamma: 1.0
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.183.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.184.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_pre_cmhc.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.185.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post_backward.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:19.186.000 [mindspore/ops/operations/_custom_ops_utils.py:665] Cannot find file [aclnn_mhc_post.h] in paths [/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_ascendc_910b/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicore_ops/op_api/include/,/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/include/,/usr/local/Ascend/cann-9.0.0/opp/../include/aclnnop]
2026-07-14 06:47:19,201 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/parallel_core/training_graph/transformer/utils.py:397] - INFO - Set full recompute at layer 3
2026-07-14 06:47:21,642 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1286] - INFO - Parameters to be loaded into network on the current rank: 
 Parameter: decoder.layers.0.input_layernorm.weight                      | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.attn_hc.alpha_pre                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.linear_kv_down_proj.weight   | Shape: (576, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.core_attention.indexer.k_norm.gamma | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.experts.weight2                         | Shape: (65536, 3584)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.attn_hc.alpha_post                          | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.ffn_hc.bias_res                             | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.router.seed                             | Shape: (1,)               | Requires Grad: False | Dtype: Int64
 Parameter: decoder.layers.0.self_attention.linear_kv_down_proj.weight   | Shape: (576, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.linear_proj.weight           | Shape: (3584, 4096)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.attn_hc.bias_res                            | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.core_attention.indexer.linear_weights_proj.weight | Shape: (64, 3584)         | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.attn_hc.alpha_pre                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.ffn_hc.mapping_weight                       | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.linear_proj.weight           | Shape: (3584, 4096)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.linear_kv_down_proj.weight   | Shape: (576, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.linear_kv_up_proj.weight     | Shape: (8192, 512)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.pre_mlp_layernorm.weight                    | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.q_layernorm.weight           | Shape: (768,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.core_attention.indexer.k_norm.gamma | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.core_attention.indexer.linear_wq_b.weight | Shape: (8192, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.core_attention.indexer.linear_wk.weight | Shape: (128, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.attn_hc.alpha_res                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.attn_hc.mapping_weight                      | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.attn_hc.bias_res                            | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.router.expert_bias                      | Shape: (64,)              | Requires Grad: False | Dtype: Float32
 Parameter: decoder.layers.1.ffn_hc.bias_pre                             | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.linear_q_up_proj.weight      | Shape: (6144, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.attn_hc.bias_res                            | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.input_layernorm.weight                      | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.shared_experts.linear_fc2.weight        | Shape: (3584, 1024)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.experts.weight1                         | Shape: (229376, 2048)     | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.attn_hc.alpha_res                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.router.fi_accu                          | Shape: (64,)              | Requires Grad: False | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.core_attention.indexer.linear_wq_b.weight | Shape: (8192, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.attn_hc.alpha_res                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.ffn_hc.bias_post                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.linear_q_down_proj.weight    | Shape: (768, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.linear_q_up_proj.weight      | Shape: (6144, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.attn_hc.bias_res                            | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.q_layernorm.weight           | Shape: (768,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.router.seed                             | Shape: (1,)               | Requires Grad: False | Dtype: Int64
 Parameter: decoder.layers.0.ffn_hc.alpha_post                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.attn_hc.alpha_res                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.linear_q_up_proj.weight      | Shape: (6144, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.router.weight                           | Shape: (64, 3584)         | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.shared_experts.linear_fc2.weight        | Shape: (3584, 1024)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.pre_mlp_layernorm.weight                    | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.mlp.linear_fc2.weight                       | Shape: (3584, 9216)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.ffn_hc.alpha_pre                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.linear_kv_up_proj.weight     | Shape: (8192, 512)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.ffn_hc.alpha_pre                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.core_attention.indexer.linear_wk.weight | Shape: (128, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.ffn_hc.mapping_weight                       | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.ffn_hc.bias_pre                             | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: output_layer.weight                                          | Shape: (131072, 3584)     | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.ffn_hc.alpha_pre                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.attn_hc.bias_pre                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.kv_layernorm.weight          | Shape: (512,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.ffn_hc.bias_res                             | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: embedding.word_embeddings.weight                             | Shape: (131072, 3584)     | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.core_attention.indexer.linear_wk.weight | Shape: (128, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.kv_layernorm.weight          | Shape: (512,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.experts.weight2                         | Shape: (65536, 3584)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.ffn_hc.mapping_weight                       | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.ffn_hc.alpha_post                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.linear_proj.weight           | Shape: (3584, 4096)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.kv_layernorm.weight          | Shape: (512,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.ffn_hc.alpha_post                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.core_attention.indexer.k_norm.beta | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.attn_hc.bias_pre                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.ffn_hc.alpha_post                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.router.offset                           | Shape: (1,)               | Requires Grad: False | Dtype: Int64
 Parameter: decoder.layers.3.ffn_hc.bias_post                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.attn_hc.bias_pre                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.ffn_hc.bias_res                             | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.router.weight                           | Shape: (64, 3584)         | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.core_attention.indexer.linear_wq_b.weight | Shape: (8192, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.linear_q_down_proj.weight    | Shape: (768, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.ffn_hc.bias_post                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.attn_hc.mapping_weight                      | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.ffn_hc.alpha_res                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.linear_proj.weight           | Shape: (3584, 4096)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.ffn_hc.alpha_res                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.linear_kv_down_proj.weight   | Shape: (576, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.shared_experts.linear_fc1.weight        | Shape: (2048, 3584)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.attn_hc.bias_pre                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.ffn_hc.bias_res                             | Shape: (1, 1, 1, 24)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.router.fi_accu                          | Shape: (64,)              | Requires Grad: False | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.linear_q_up_proj.weight      | Shape: (6144, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.input_layernorm.weight                      | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.core_attention.indexer.linear_weights_proj.weight | Shape: (64, 3584)         | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.core_attention.indexer.linear_wk.weight | Shape: (128, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.pre_mlp_layernorm.weight                    | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.core_attention.indexer.k_norm.beta | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.linear_q_down_proj.weight    | Shape: (768, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.ffn_hc.bias_post                            | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.core_attention.indexer.k_norm.gamma | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.attn_hc.mapping_weight                      | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.attn_hc.mapping_weight                      | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.ffn_hc.mapping_weight                       | Shape: (14336, 32)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.attn_hc.bias_post                           | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.pre_mlp_layernorm.weight                    | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.core_attention.indexer.linear_wq_b.weight | Shape: (8192, 768)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.experts.weight1                         | Shape: (229376, 2048)     | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.ffn_hc.bias_pre                             | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.input_layernorm.weight                      | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.core_attention.indexer.linear_weights_proj.weight | Shape: (64, 3584)         | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.mlp.router.offset                           | Shape: (1,)               | Requires Grad: False | Dtype: Int64
 Parameter: decoder.layers.2.self_attention.linear_q_down_proj.weight    | Shape: (768, 3584)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.ffn_hc.alpha_pre                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.linear_kv_up_proj.weight     | Shape: (8192, 512)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.core_attention.indexer.k_norm.gamma | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.attn_hc.alpha_post                          | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.self_attention.core_attention.indexer.k_norm.beta | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.ffn_hc.alpha_res                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.self_attention.kv_layernorm.weight          | Shape: (512,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.attn_hc.bias_post                           | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.ffn_hc.alpha_res                            | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.mlp.linear_fc2.weight                       | Shape: (3584, 9216)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.q_layernorm.weight           | Shape: (768,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.attn_hc.bias_post                           | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.linear_kv_up_proj.weight     | Shape: (8192, 512)        | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.attn_hc.alpha_pre                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.attn_hc.alpha_post                          | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.router.expert_bias                      | Shape: (64,)              | Requires Grad: False | Dtype: Float32
 Parameter: decoder.final_layernorm.weight                               | Shape: (3584,)            | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.ffn_hc.bias_pre                             | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.q_layernorm.weight           | Shape: (768,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.self_attention.core_attention.indexer.k_norm.beta | Shape: (128,)             | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.1.mlp.linear_fc1.weight                       | Shape: (18432, 3584)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.mlp.linear_fc1.weight                       | Shape: (18432, 3584)      | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.self_attention.core_attention.indexer.linear_weights_proj.weight | Shape: (64, 3584)         | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.0.attn_hc.alpha_post                          | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.2.attn_hc.bias_post                           | Shape: (1, 1, 1, 4)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.mlp.shared_experts.linear_fc1.weight        | Shape: (2048, 3584)       | Requires Grad: True  | Dtype: Float32
 Parameter: decoder.layers.3.attn_hc.alpha_pre                           | Shape: (1, 1, 1, 1)       | Requires Grad: True  | Dtype: Float32
2026-07-14 06:47:21,646 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:802] - INFO - Network Parameters: 2714 M.
2026-07-14 06:47:21,647 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1318] - INFO - .........Build Optimizer For Train..........
2026-07-14 06:47:21,647 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:629] - INFO - ..........Build Optimizer From Config..........
2026-07-14 06:47:21,647 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:693] - INFO - ..........Build LR Schedule From Config..........
2026-07-14 06:47:21,649 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/optimizer_grouped_parameters.py:181] - INFO - Param groups = {
  "weight_decay": {
    "weight_decay": 0.1,
    "params": [
      "embedding.word_embeddings.weight",
      "decoder.layers.0.self_attention.core_attention.indexer.linear_wq_b.weight",
      "decoder.layers.0.self_attention.core_attention.indexer.linear_wk.weight",
      "decoder.layers.0.self_attention.core_attention.indexer.linear_weights_proj.weight",
      "decoder.layers.0.self_attention.linear_proj.weight",
      "decoder.layers.0.self_attention.linear_q_down_proj.weight",
      "decoder.layers.0.self_attention.linear_q_up_proj.weight",
      "decoder.layers.0.self_attention.linear_kv_down_proj.weight",
      "decoder.layers.0.self_attention.linear_kv_up_proj.weight",
      "decoder.layers.0.mlp.linear_fc1.weight",
      "decoder.layers.0.mlp.linear_fc2.weight",
      "decoder.layers.0.attn_hc.mapping_weight",
      "decoder.layers.0.attn_hc.alpha_pre",
      "decoder.layers.0.attn_hc.alpha_post",
      "decoder.layers.0.attn_hc.alpha_res",
      "decoder.layers.0.attn_hc.bias_pre",
      "decoder.layers.0.attn_hc.bias_post",
      "decoder.layers.0.attn_hc.bias_res",
      "decoder.layers.0.ffn_hc.mapping_weight",
      "decoder.layers.0.ffn_hc.alpha_pre",
      "decoder.layers.0.ffn_hc.alpha_post",
      "decoder.layers.0.ffn_hc.alpha_res",
      "decoder.layers.0.ffn_hc.bias_pre",
      "decoder.layers.0.ffn_hc.bias_post",
      "decoder.layers.0.ffn_hc.bias_res",
      "decoder.layers.1.self_attention.core_attention.indexer.linear_wq_b.weight",
      "decoder.layers.1.self_attention.core_attention.indexer.linear_wk.weight",
      "decoder.layers.1.self_attention.core_attention.indexer.linear_weights_proj.weight",
      "decoder.layers.1.self_attention.linear_proj.weight",
      "decoder.layers.1.self_attention.linear_q_down_proj.weight",
      "decoder.layers.1.self_attention.linear_q_up_proj.weight",
      "decoder.layers.1.self_attention.linear_kv_down_proj.weight",
      "decoder.layers.1.self_attention.linear_kv_up_proj.weight",
      "decoder.layers.1.mlp.linear_fc1.weight",
      "decoder.layers.1.mlp.linear_fc2.weight",
      "decoder.layers.1.attn_hc.mapping_weight",
      "decoder.layers.1.attn_hc.alpha_pre",
      "decoder.layers.1.attn_hc.alpha_post",
      "decoder.layers.1.attn_hc.alpha_res",
      "decoder.layers.1.attn_hc.bias_pre",
      "decoder.layers.1.attn_hc.bias_post",
      "decoder.layers.1.attn_hc.bias_res",
      "decoder.layers.1.ffn_hc.mapping_weight",
      "decoder.layers.1.ffn_hc.alpha_pre",
      "decoder.layers.1.ffn_hc.alpha_post",
      "decoder.layers.1.ffn_hc.alpha_res",
      "decoder.layers.1.ffn_hc.bias_pre",
      "decoder.layers.1.ffn_hc.bias_post",
      "decoder.layers.1.ffn_hc.bias_res",
      "decoder.layers.2.self_attention.core_attention.indexer.linear_wq_b.weight",
      "decoder.layers.2.self_attention.core_attention.indexer.linear_wk.weight",
      "decoder.layers.2.self_attention.core_attention.indexer.linear_weights_proj.weight",
      "decoder.layers.2.self_attention.linear_proj.weight",
      "decoder.layers.2.self_attention.linear_q_down_proj.weight",
      "decoder.layers.2.self_attention.linear_q_up_proj.weight",
      "decoder.layers.2.self_attention.linear_kv_down_proj.weight",
      "decoder.layers.2.self_attention.linear_kv_up_proj.weight",
      "decoder.layers.2.mlp.router.weight",
      "decoder.layers.2.mlp.experts.weight1",
      "decoder.layers.2.mlp.experts.weight2",
      "decoder.layers.2.mlp.shared_experts.linear_fc1.weight",
      "decoder.layers.2.mlp.shared_experts.linear_fc2.weight",
      "decoder.layers.2.attn_hc.mapping_weight",
      "decoder.layers.2.attn_hc.alpha_pre",
      "decoder.layers.2.attn_hc.alpha_post",
      "decoder.layers.2.attn_hc.alpha_res",
      "decoder.layers.2.attn_hc.bias_pre",
      "decoder.layers.2.attn_hc.bias_post",
      "decoder.layers.2.attn_hc.bias_res",
      "decoder.layers.2.ffn_hc.mapping_weight",
      "decoder.layers.2.ffn_hc.alpha_pre",
      "decoder.layers.2.ffn_hc.alpha_post",
      "decoder.layers.2.ffn_hc.alpha_res",
      "decoder.layers.2.ffn_hc.bias_pre",
      "decoder.layers.2.ffn_hc.bias_post",
      "decoder.layers.2.ffn_hc.bias_res",
      "decoder.layers.3.self_attention.core_attention.indexer.linear_wq_b.weight",
      "decoder.layers.3.self_attention.core_attention.indexer.linear_wk.weight",
      "decoder.layers.3.self_attention.core_attention.indexer.linear_weights_proj.weight",
      "decoder.layers.3.self_attention.linear_proj.weight",
      "decoder.layers.3.self_attention.linear_q_down_proj.weight",
      "decoder.layers.3.self_attention.linear_q_up_proj.weight",
      "decoder.layers.3.self_attention.linear_kv_down_proj.weight",
      "decoder.layers.3.self_attention.linear_kv_up_proj.weight",
      "decoder.layers.3.mlp.router.weight",
      "decoder.layers.3.mlp.experts.weight1",
      "decoder.layers.3.mlp.experts.weight2",
      "decoder.layers.3.mlp.shared_experts.linear_fc1.weight",
      "decoder.layers.3.mlp.shared_experts.linear_fc2.weight",
      "decoder.layers.3.attn_hc.mapping_weight",
      "decoder.layers.3.attn_hc.alpha_pre",
      "decoder.layers.3.attn_hc.alpha_post",
      "decoder.layers.3.attn_hc.alpha_res",
      "decoder.layers.3.attn_hc.bias_pre",
      "decoder.layers.3.attn_hc.bias_post",
      "decoder.layers.3.attn_hc.bias_res",
      "decoder.layers.3.ffn_hc.mapping_weight",
      "decoder.layers.3.ffn_hc.alpha_pre",
      "decoder.layers.3.ffn_hc.alpha_post",
      "decoder.layers.3.ffn_hc.alpha_res",
      "decoder.layers.3.ffn_hc.bias_pre",
      "decoder.layers.3.ffn_hc.bias_post",
      "decoder.layers.3.ffn_hc.bias_res",
      "output_layer.weight"
    ]
  },
  "no_weight_decay": {
    "weight_decay": 0.0,
    "params": [
      "decoder.layers.0.input_layernorm.weight",
      "decoder.layers.0.self_attention.core_attention.indexer.k_norm.gamma",
      "decoder.layers.0.self_attention.core_attention.indexer.k_norm.beta",
      "decoder.layers.0.self_attention.q_layernorm.weight",
      "decoder.layers.0.self_attention.kv_layernorm.weight",
      "decoder.layers.0.pre_mlp_layernorm.weight",
      "decoder.layers.1.input_layernorm.weight",
      "decoder.layers.1.self_attention.core_attention.indexer.k_norm.gamma",
      "decoder.layers.1.self_attention.core_attention.indexer.k_norm.beta",
      "decoder.layers.1.self_attention.q_layernorm.weight",
      "decoder.layers.1.self_attention.kv_layernorm.weight",
      "decoder.layers.1.pre_mlp_layernorm.weight",
      "decoder.layers.2.input_layernorm.weight",
      "decoder.layers.2.self_attention.core_attention.indexer.k_norm.gamma",
      "decoder.layers.2.self_attention.core_attention.indexer.k_norm.beta",
      "decoder.layers.2.self_attention.q_layernorm.weight",
      "decoder.layers.2.self_attention.kv_layernorm.weight",
      "decoder.layers.2.pre_mlp_layernorm.weight",
      "decoder.layers.3.input_layernorm.weight",
      "decoder.layers.3.self_attention.core_attention.indexer.k_norm.gamma",
      "decoder.layers.3.self_attention.core_attention.indexer.k_norm.beta",
      "decoder.layers.3.self_attention.q_layernorm.weight",
      "decoder.layers.3.self_attention.kv_layernorm.weight",
      "decoder.layers.3.pre_mlp_layernorm.weight",
      "decoder.final_layernorm.weight"
    ]
  }
}
2026-07-14 06:47:21,649 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/core/optim/build_optim.py:40] - INFO - tft is not valid
2026-07-14 06:47:21,780 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1326] - INFO - .........Build Running Wrapper From Config For Train..........
2026-07-14 06:47:21,781 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:740] - INFO - .........Build Model Wrapper for Train From Config..........
2026-07-14 06:47:21,786 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1343] - INFO - .........Build Callbacks For Train..........
2026-07-14 06:47:21,788 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1395] - INFO - Recommend using weights in the safetensors format.
2026-07-14 06:47:21,788 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1521] - INFO - .........Starting Init Train Model..........
{'adapter_id': None,
 'auto_trans_ckpt': False,
 'balanced_load': False,
 'callbacks': [{'per_print_times': 1, 'type': 'MFLossMonitor'},
               {'balance_via_topk_bias': True,
                'expert_num': 64,
                'micro_batch_num': 1,
                'topk_bias_update_rate': 0.0001,
                'type': 'TopkBiasBalanceCallback'},
               {'checkpoint_format': 'safetensors',
                'keep_checkpoint_max': 3,
                'prefix': 'telechat4',
                'remove_redundancy': False,
                'save_checkpoint_steps': 1000,
                'type': 'CheckpointMonitor'},
               {'stop_step': 30, 'type': 'TrainCallBack'}],
 'checkpoint': {'async_save': False,
                'load_balanced': False,
                'load_path': None,
                'no_load_optim': True,
                'no_save_optim': False,
                'prefix': 'telechat4',
                'reshard_worker_num': 1,
                'save_interleaved_steps': 1000,
                'save_max': 3,
                'save_path': './output/checkpoint',
                'save_remove_redundancy': False},
 'context': {'affinity_cpu_list': False,
             'ascend_config': {'parallel_speed_up_json_path': '/lgsl_data/dianxin_telechat/mindformers/parallel_speed_up_64M.json'},
             'device_id': 8,
             'device_target': 'Ascend',
             'jit_config': {'jit_level': 'O1'},
             'max_call_depth': 10000,
             'max_device_memory': '54GB',
             'memory_optimize_level': 'O0',
             'mempool_block_size': '55GB',
             'mode': 0,
             'save_graphs': True,
             'save_graphs_path': './graph'},
 'data_size': 1,
 'data_skip_steps': None,
 'device_num': 1,
 'do_eval': False,
 'eval_epoch_interval': -1,
 'eval_step_interval': 100,
 'exclude_cann_cpu': False,
 'ignore_data_skip': False,
 'infer_precision_sync': None,
 'init_start_profile': False,
 'input_data': None,
 'layer_decay': 0.65,
 'layer_scale': False,
 'load_checkpoint': '',
 'load_ckpt_async': False,
 'load_ckpt_format': 'safetensors',
 'local_rank': 0,
 'lr_scale': False,
 'lr_scale_factor': 256,
 'lr_schedule': {'learning_rate': 0.0002,
                 'lr_end': 1e-05,
                 'total_steps': 6,
                 'type': 'CosineWithWarmUpLR',
                 'warmup_lr_init': 0,
                 'warmup_steps': 200},
 'metric': [{'type': 'PerplexityMetric'}],
 'micro_batch_interleave_num': 1,
 'model': {'model_config': {'add_bias_linear': False,
                            'architectures': ['TeleChat4MoeForCausalLM'],
                            'attention_dropout': 0.0,
                            'beta_fast': 32,
                            'beta_slow': 1,
                            'checkpoint_name_or_path': None,
                            'cmhc_gamma': 0.8,
                            'cmhc_gamma_num_layers': 2,
                            'compute_dtype': 'bfloat16',
                            'disable_lazy_inline': True,
                            'dsa_indexer_head_dim': 128,
                            'dsa_indexer_loss_coeff': 1.0,
                            'dsa_indexer_n_heads': 64,
                            'dsa_indexer_topk': 2048,
                            'dsa_indexer_use_sparse_loss': True,
                            'enable_cmhc': True,
                            'enable_fused_mhc_ops': True,
                            'enable_fused_triton_sinkhorn': False,
                            'enable_hyper_connections': True,
                            'experimental_attention_variant': 'dsa',
                            'first_k_dense_replace': 2,
                            'gated_linear_unit': True,
                            'gradient_accumulation_steps': 1,
                            'hc_sinkhorn_clamp': 30.0,
                            'hidden_act': 'silu',
                            'hidden_dropout': 0.0,
                            'hidden_size': 3584,
                            'initializer_range': 0.02,
                            'input_sliced_sig': True,
                            'intermediate_size': 9216,
                            'kv_lora_rank': 512,
                            'layernorm_compute_dtype': 'float32',
                            'max_position_embeddings': 4096,
                            'mhc_init_gating_factor': 0.01,
                            'mhc_sinkhorn_iterations': 20,
                            'mla_qkv_concat': False,
                            'model_type': 'telechat4_moe',
                            'moe_aux_loss_coeff': 0.0008,
                            'moe_grouped_gemm': True,
                            'moe_intermediate_size': 1024,
                            'moe_router_bias_update_rate': 0.0001,
                            'moe_router_dtype': 'float32',
                            'moe_router_enable_expert_bias': True,
                            'moe_router_force_expert_balance': False,
                            'moe_router_load_balancing_type': 'seq_aux_loss',
                            'moe_shared_expert_intermediate_size': 1024,
                            'moe_token_dispatcher_type': 'alltoall',
                            'mscale': 1,
                            'mscale_all_dim': 1,
                            'mtp_loss_factor': 0.0,
                            'multi_latent_attention': True,
                            'n_group': None,
                            'n_routed_experts': 64,
                            'n_shared_experts': 1,
                            'norm_topk_prob': True,
                            'num_attention_heads': 32,
                            'num_experts_per_tok': 4,
                            'num_hidden_layers': 4,
                            'num_nextn_predict_layers': 0,
                            'num_residual_streams': 4,
                            'params_dtype': 'float32',
                            'position_embedding_type': 'rope',
                            'pp_interleave_num': 1,
                            'q_lora_rank': 768,
                            'qk_layernorm': True,
                            'qk_nope_head_dim': 128,
                            'qk_rope_head_dim': 64,
                            'rms_norm_eps': 1e-06,
                            'rope_theta': 10000.0,
                            'rotary_dtype': 'float32',
                            'rotary_scaling_factor': 1.0,
                            'routed_scaling_factor': 2.0,
                            'router_jitter_noise': 0.01,
                            'scoring_func': 'sigmoid',
                            'seq_length': 4096,
                            'softmax_compute_dtype': 'float32',
                            'topk_group': None,
                            'use_flash_attention': True,
                            'use_pad_tokens': True,
                            'v_head_dim': 128,
                            'vocab_size': 131072}},
 'moe_config': {},
 'monitor_config': {'check_for_global_norm': False,
                    'device_local_loss_format': None,
                    'device_local_norm_format': None,
                    'dump_path': './dump',
                    'global_norm_spike_count_threshold': 10,
                    'global_norm_spike_threshold': 3.0,
                    'hc_res_stats_format': None,
                    'health_checkpoint': None,
                    'invert': False,
                    'local_loss_format': None,
                    'local_norm_format': None,
                    'max_attention_logit_format': None,
                    'monitor_on': False,
                    'optimizer_state_format': None,
                    'print_struct': False,
                    'stable_rank_config': <mindformers.tools.register.template.StableRankConfig object at 0xfffe975aa1d0>,
                    'step_interval': 1,
                    'target': ['.*'],
                    'throughput_baseline': None,
                    'weight_state_format': None},
 'mstx': False,
 'only_save_strategy': False,
 'optimizer': {'betas': [0.9, 0.95],
               'eps': 1e-08,
               'learning_rate': CosineWithWarmUpLR(),
               'params': [{'params': [Parameter (name=embedding.word_embeddings.weight, shape=(131072, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.core_attention.indexer.linear_wq_b.weight, shape=(8192, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.core_attention.indexer.linear_wk.weight, shape=(128, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.core_attention.indexer.linear_weights_proj.weight, shape=(64, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.linear_proj.weight, shape=(3584, 4096), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.linear_q_down_proj.weight, shape=(768, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.linear_q_up_proj.weight, shape=(6144, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.linear_kv_down_proj.weight, shape=(576, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.linear_kv_up_proj.weight, shape=(8192, 512), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.mlp.linear_fc1.weight, shape=(18432, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.mlp.linear_fc2.weight, shape=(3584, 9216), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.attn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.attn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.attn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.attn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.attn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.attn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.attn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.ffn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.ffn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.ffn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.ffn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.ffn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.ffn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.ffn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.core_attention.indexer.linear_wq_b.weight, shape=(8192, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.core_attention.indexer.linear_wk.weight, shape=(128, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.core_attention.indexer.linear_weights_proj.weight, shape=(64, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.linear_proj.weight, shape=(3584, 4096), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.linear_q_down_proj.weight, shape=(768, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.linear_q_up_proj.weight, shape=(6144, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.linear_kv_down_proj.weight, shape=(576, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.linear_kv_up_proj.weight, shape=(8192, 512), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.mlp.linear_fc1.weight, shape=(18432, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.mlp.linear_fc2.weight, shape=(3584, 9216), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.attn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.attn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.attn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.attn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.attn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.attn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.attn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.ffn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.ffn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.ffn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.ffn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.ffn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.ffn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.ffn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.core_attention.indexer.linear_wq_b.weight, shape=(8192, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.core_attention.indexer.linear_wk.weight, shape=(128, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.core_attention.indexer.linear_weights_proj.weight, shape=(64, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.linear_proj.weight, shape=(3584, 4096), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.linear_q_down_proj.weight, shape=(768, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.linear_q_up_proj.weight, shape=(6144, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.linear_kv_down_proj.weight, shape=(576, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.linear_kv_up_proj.weight, shape=(8192, 512), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.mlp.router.weight, shape=(64, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.mlp.experts.weight1, shape=(229376, 2048), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.mlp.experts.weight2, shape=(65536, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.mlp.shared_experts.linear_fc1.weight, shape=(2048, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.mlp.shared_experts.linear_fc2.weight, shape=(3584, 1024), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.attn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.attn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.attn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.attn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.attn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.attn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.attn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.ffn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.ffn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.ffn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.ffn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.ffn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.ffn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.ffn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.core_attention.indexer.linear_wq_b.weight, shape=(8192, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.core_attention.indexer.linear_wk.weight, shape=(128, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.core_attention.indexer.linear_weights_proj.weight, shape=(64, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.linear_proj.weight, shape=(3584, 4096), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.linear_q_down_proj.weight, shape=(768, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.linear_q_up_proj.weight, shape=(6144, 768), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.linear_kv_down_proj.weight, shape=(576, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.linear_kv_up_proj.weight, shape=(8192, 512), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.mlp.router.weight, shape=(64, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.mlp.experts.weight1, shape=(229376, 2048), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.mlp.experts.weight2, shape=(65536, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.mlp.shared_experts.linear_fc1.weight, shape=(2048, 3584), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.mlp.shared_experts.linear_fc2.weight, shape=(3584, 1024), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.attn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.attn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.attn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.attn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.attn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.attn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.attn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.ffn_hc.mapping_weight, shape=(14336, 32), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.ffn_hc.alpha_pre, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.ffn_hc.alpha_post, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.ffn_hc.alpha_res, shape=(1, 1, 1, 1), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.ffn_hc.bias_pre, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.ffn_hc.bias_post, shape=(1, 1, 1, 4), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.ffn_hc.bias_res, shape=(1, 1, 1, 24), dtype=Float32, requires_grad=True),
                                      Parameter (name=output_layer.weight, shape=(131072, 3584), dtype=Float32, requires_grad=True)],
                           'weight_decay': 0.1},
                          {'params': [Parameter (name=decoder.layers.0.input_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.core_attention.indexer.k_norm.gamma, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.core_attention.indexer.k_norm.beta, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.q_layernorm.weight, shape=(768,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.self_attention.kv_layernorm.weight, shape=(512,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.0.pre_mlp_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.input_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.core_attention.indexer.k_norm.gamma, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.core_attention.indexer.k_norm.beta, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.q_layernorm.weight, shape=(768,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.self_attention.kv_layernorm.weight, shape=(512,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.1.pre_mlp_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.input_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.core_attention.indexer.k_norm.gamma, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.core_attention.indexer.k_norm.beta, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.q_layernorm.weight, shape=(768,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.self_attention.kv_layernorm.weight, shape=(512,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.2.pre_mlp_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.input_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.core_attention.indexer.k_norm.gamma, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.core_attention.indexer.k_norm.beta, shape=(128,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.q_layernorm.weight, shape=(768,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.self_attention.kv_layernorm.weight, shape=(512,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.layers.3.pre_mlp_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True),
                                      Parameter (name=decoder.final_layernorm.weight, shape=(3584,), dtype=Float32, requires_grad=True)],
                           'weight_decay': 0.0}],
               'type': 'AdamW',
               'weight_decay': 0.1},
 'output_dir': './output',
 'parallel': {'dataset_strategy': ((1, 1), (1, 1), (1, 1), (1, 1)),
              'enable_alltoall': True,
              'enable_parallel_optimizer': None,
              'full_batch': False,
              'gradients_mean': False,
              'parallel_mode': 1,
              'parallel_optimizer_config': {'gradient_accumulation_shard': False,
                                            'optimizer_weight_shard_size': 1,
                                            'parallel_optimizer_threshold': 64},
              'pipeline_config': {'pipeline_interleave': False,
                                  'pipeline_scheduler': '1f1b'},
              'search_mode': 'sharding_propagation',
              'strategy_ckpt_config': {'only_trainable_params': False,
                                       'save_file': './ckpt_strategy.ckpt'},
              'strategy_ckpt_save_file': '/lgsl_data/dianxin_telechat/mindformers/output/strategy/ckpt_strategy_rank_0.ckpt'},
 'parallel_config': <mindformers.modules.transformer.transformer.TransformerOpParallelConfig object at 0xfffddcc17b50>,
 'postprocess_use_numpy': False,
 'predict_batch_size': None,
 'pretrained_model_dir': '',
 'print_separate_loss': True,
 'profile': False,
 'profile_communication': False,
 'profile_level': 1,
 'profile_memory': True,
 'profile_output': '/hpfs/huawei/zwj/dsa/profile_0611/',
 'profile_pipeline': False,
 'profile_rank_ids': None,
 'profile_start_step': 3,
 'profile_stop_step': 6,
 'rank_id': 0,
 'recompute_config': <mindformers.modules.transformer.transformer.TransformerRecomputeConfig object at 0xfffddcd9f1d0>,
 'remote_save_url': None,
 'remove_redundancy': False,
 'reshard_worker_num': 1,
 'resume_training': False,
 'run_mode': 'train',
 'runner_config': {'batch_size': 1,
                   'epochs': 6,
                   'global_batch_size': 1,
                   'gradient_accumulation_steps': 1,
                   'initial_epoch': 0,
                   'initial_step': 0,
                   'mini_batch_size': 1,
                   'num_classes': 1,
                   'origin_epochs': 6,
                   'sink_mode': True,
                   'sink_size': 1,
                   'stop_step': 0},
 'runner_wrapper': {'scale_sense': 1.0,
                    'type': 'MFTrainOneStepCell',
                    'use_clip_grad': True},
 'save_file': None,
 'seed': 1234,
 'src_strategy_path_or_dir': '',
 'swap_config': <mindformers.modules.transformer.transformer.TransformerSwapConfig object at 0xfffddccae850>,
 'train_dataset': {'auto_tune': None,
                   'autotune_per_step': None,
                   'batch_size': 1,
                   'data_loader': {'compressed_eod_mask_length': 128,
                                   'create_attention_mask': False,
                                   'create_compressed_eod_mask': False,
                                   'create_seq_len_vector': False,
                                   'data_path': '/hpfs/huawei/lql/dsa/single-417/single',
                                   'eod_mask_loss': False,
                                   'eod_token': 2,
                                   'pad_token': 3,
                                   'reset_attention_mask': False,
                                   'reset_position_ids': False,
                                   'sequence_length': 4096,
                                   'shuffle': False,
                                   'type': 'OrderedIndexDataLoader'},
                   'do_eval': False,
                   'drop_remainder': True,
                   'eod_reset': False,
                   'filepath_prefix': None,
                   'input_columns': ['input_ids',
                                     'labels',
                                     'loss_mask',
                                     'position_ids'],
                   'micro_batch_num': 1,
                   'num_parallel_workers': 8,
                   'numa_enable': False,
                   'output_columns': ['input_ids',
                                      'labels',
                                      'loss_mask',
                                      'position_ids'],
                   'prefetch_size': 1,
                   'profile': False,
                   'python_multiprocessing': False,
                   'repeat': 1,
                   'seed': 1234},
 'train_dataset_task': {'dataset_config': {'auto_tune': None,
                                           'autotune_per_step': None,
                                           'batch_size': 1,
                                           'data_loader': {'compressed_eod_mask_length': 128,
                                                           'create_attention_mask': False,
                                                           'create_compressed_eod_mask': False,
                                                           'create_seq_len_vector': False,
                                                           'data_path': '/hpfs/huawei/lql/dsa/single-417/single',
                                                           'eod_mask_loss': False,
                                                           'eod_token': 2,
                                                           'pad_token': 3,
                                                           'reset_attention_mask': False,
                                                           'reset_position_ids': False,
                                                           'sequence_length': 4096,
                                                           'shuffle': False,
                                                           'type': 'OrderedIndexDataLoader'},
                                           'do_eval': False,
                                           'drop_remainder': True,
                                           'eod_reset': False,
                                           'filepath_prefix': None,
                                           'input_columns': ['input_ids',
                                                             'labels',
                                                             'loss_mask',
                                                             'position_ids'],
                                           'micro_batch_num': 1,
                                           'num_parallel_workers': 8,
                                           'numa_enable': False,
                                           'output_columns': ['input_ids',
                                                              'labels',
                                                              'loss_mask',
                                                              'position_ids'],
                                           'prefetch_size': 1,
                                           'profile': False,
                                           'python_multiprocessing': False,
                                           'repeat': 1,
                                           'seed': 1234},
                        'type': 'CausalLanguageModelDataset'},
 'train_precision_sync': True,
 'trainer': {'model_name': 'telechat4_moe',
             'type': 'CausalLanguageModelingTrainer'},
 'transform_process_num': 1,
 'use_legacy': False,
 'use_legacy_format': True,
 'use_parallel': False}
2026-07-14 06:47:21,815 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1679] - INFO - .........Model Compiling, Please Wait a Moment...........
2026-07-14 06:47:21,815 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/trainer/base_trainer.py:1686] - INFO - .........Starting Training Model..........
[WARNING] ME(56203:281473441098720,MainProcess):2026-07-14-06:47:21.816.000 [mindspore/train/model.py:1403] For CheckpointMonitor callback, {'end', 'step_end'} methods may not be supported in later version, Use methods prefixed with 'on_train' or 'on_eval' instead when using customized callbacks.
2026-07-14 06:47:21,819 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
2026-07-14 06:47:21,821 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
2026-07-14 06:47:23,567 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
2026-07-14 06:47:23,664 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
[WARNING] ME(56203:281473183248800,MainProcess):2026-07-14-06:47:23.764.000 [mindspore/dataset/engine/datasets_user_defined.py:693] idx_queue is not empty
2026-07-14 06:47:23,793 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
2026-07-14 06:47:23,890 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
[WARNING] ME(56203:281473183248800,MainProcess):2026-07-14-06:47:23.981.000 [mindspore/dataset/engine/datasets_user_defined.py:693] idx_queue is not empty
2026-07-14 06:47:23,986 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
2026-07-14 06:47:24,097 - mindformers/lgsl_data/dianxin_telechat/mindformers/output/log[mindformers/dataset/dataloader/ordered_index_dataloader.py:496] - WARNING - idx: 0 data got 4096 shorter than the configured sequence length 4096, thus data will be padded to 4096 + 1 with pad token: 3.
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.855.906 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@363_2↓mindformers_parallel_core_training_graph_base_models_gpt_gpt_model_GPTModel_construct_4370:loss_mask{[0]: ValueNode<FuncGraph> 2125__tuple_getitem_by_number_352, [1]: CNode_349, [2]: ValueNode<Int64Imm> 2}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.855.973 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@363_2↓mindformers_parallel_core_training_graph_base_models_gpt_gpt_model_GPTModel_construct_4370:rotary_pos_emb{[0]: ValueNode<FuncGraph> 2126__tuple_getitem_by_number_4348, [1]: CNode_4346, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.855.995 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@363_2↓mindformers_parallel_core_training_graph_base_models_gpt_gpt_model_GPTModel_construct_4370:labels{[0]: ValueNode<FuncGraph> 2127__tuple_getitem_by_number_4349, [1]: CNode_349, [2]: ValueNode<Int64Imm> 0}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.865.105 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@369_5↓mindformers_parallel_core_training_graph_base_models_gpt_gpt_model_GPTModel_construct_4427:_{[0]: ValueNode<FuncGraph> 2113__tuple_getitem_by_number_4371, [1]: CNode_4369, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.865.725 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@428_↓↻↓mindformers_parallel_core_training_graph_transformer_transformer_block_TransformerBlock_construct_4211:context{[0]: ValueNode<FuncGraph> 1706__tuple_getitem_by_number_1284, [1]: CNode_1282, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.866.060 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@435_↓↻↓mindformers_parallel_core_training_graph_transformer_transformer_block_TransformerBlock_construct_4198:context{[0]: ValueNode<FuncGraph> 1343__tuple_getitem_by_number_2132, [1]: CNode_2130, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.866.620 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@441_↓↻↓mindformers_parallel_core_training_graph_transformer_transformer_block_TransformerBlock_construct_4185:context{[0]: ValueNode<FuncGraph> 913__tuple_getitem_by_number_3143, [1]: CNode_3141, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.867.678 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@447_↓↻↓mindformers_parallel_core_training_graph_transformer_transformer_block_TransformerBlock_construct_4172:context{[0]: ValueNode<FuncGraph> 461__tuple_getitem_by_number_4147, [1]: CNode_4145, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.869.032 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1774_dsa_forward_1067:_{[0]: ValueNode<FuncGraph> 2029__tuple_getitem_by_number_1010, [1]: CNode_1009, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.869.514 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1819_get_qk_index_758:_{[0]: ValueNode<FuncGraph> 1892__tuple_getitem_by_number_546, [1]: CNode_545, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.869.537 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1819_get_qk_index_758:_{[0]: ValueNode<FuncGraph> 1893__tuple_getitem_by_number_564, [1]: CNode_563, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.869.556 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1819_get_qk_index_758:_{[0]: ValueNode<FuncGraph> 1894__tuple_getitem_by_number_582, [1]: CNode_581, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.871.364 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@940_✓↓mindformers_parallel_core_training_graph_transformer_moe_moe_layer_MoELayer_construct_2429:_{[0]: ValueNode<FuncGraph> 1034__tuple_getitem_by_number_2296, [1]: CNode_2294, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.871.841 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1927_2↓get_query_key_value_tensors_954:_{[0]: ValueNode<FuncGraph> 2008__tuple_getitem_by_number_785, [1]: CNode_783, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.871.931 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@2009_✓↓get_query_key_value_tensors_984:_{[0]: ValueNode<FuncGraph> 2013__tuple_getitem_by_number_980, [1]: CNode_977, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.872.126 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1411_dsa_forward_1931:_{[0]: ValueNode<FuncGraph> 1666__tuple_getitem_by_number_1897, [1]: CNode_1896, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.872.958 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1456_get_qk_index_1645:_{[0]: ValueNode<FuncGraph> 1529__tuple_getitem_by_number_1475, [1]: CNode_1474, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.872.982 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1456_get_qk_index_1645:_{[0]: ValueNode<FuncGraph> 1530__tuple_getitem_by_number_1493, [1]: CNode_1492, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.873.005 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1456_get_qk_index_1645:_{[0]: ValueNode<FuncGraph> 1531__tuple_getitem_by_number_1511, [1]: CNode_1510, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.873.736 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1985_✓qkv_up_proj_and_rope_apply_932:_{[0]: ValueNode<FuncGraph> 1994__tuple_getitem_by_number_928, [1]: CNode_926, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.873.900 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1780_3↓dsa_forward_1177:_{[0]: ValueNode<FuncGraph> 1797__tuple_getitem_by_number_1149, [1]: CNode_1148, [2]: ValueNode<Int64Imm> 0}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.873.924 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1780_3↓dsa_forward_1177:_{[0]: ValueNode<FuncGraph> 1798__tuple_getitem_by_number_1150, [1]: CNode_1148, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.873.944 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1780_3↓dsa_forward_1177:_{[0]: ValueNode<FuncGraph> 1799__tuple_getitem_by_number_1151, [1]: CNode_1148, [2]: ValueNode<Int64Imm> 2}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.875.489 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@488_✓↓mindformers_parallel_core_training_graph_transformer_moe_moe_layer_MoELayer_construct_3425:_{[0]: ValueNode<FuncGraph> 582__tuple_getitem_by_number_3293, [1]: CNode_3291, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.876.138 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1564_2↓get_query_key_value_tensors_1841:_{[0]: ValueNode<FuncGraph> 1645__tuple_getitem_by_number_1672, [1]: CNode_1670, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.876.224 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1646_✓↓get_query_key_value_tensors_1871:_{[0]: ValueNode<FuncGraph> 1650__tuple_getitem_by_number_1867, [1]: CNode_1864, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.876.448 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1048_dsa_forward_2942:_{[0]: ValueNode<FuncGraph> 1303__tuple_getitem_by_number_2908, [1]: CNode_2907, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.877.380 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1093_get_qk_index_2656:_{[0]: ValueNode<FuncGraph> 1166__tuple_getitem_by_number_2486, [1]: CNode_2485, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.877.403 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1093_get_qk_index_2656:_{[0]: ValueNode<FuncGraph> 1167__tuple_getitem_by_number_2504, [1]: CNode_2503, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.877.422 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1093_get_qk_index_2656:_{[0]: ValueNode<FuncGraph> 1168__tuple_getitem_by_number_2522, [1]: CNode_2521, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.877.911 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1622_✓qkv_up_proj_and_rope_apply_1819:_{[0]: ValueNode<FuncGraph> 1631__tuple_getitem_by_number_1815, [1]: CNode_1813, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.878.075 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1417_3↓dsa_forward_2020:_{[0]: ValueNode<FuncGraph> 1434__tuple_getitem_by_number_1992, [1]: CNode_1991, [2]: ValueNode<Int64Imm> 0}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.878.097 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1417_3↓dsa_forward_2020:_{[0]: ValueNode<FuncGraph> 1435__tuple_getitem_by_number_1993, [1]: CNode_1991, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.878.117 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1417_3↓dsa_forward_2020:_{[0]: ValueNode<FuncGraph> 1436__tuple_getitem_by_number_1994, [1]: CNode_1991, [2]: ValueNode<Int64Imm> 2}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.879.972 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1201_2↓get_query_key_value_tensors_2852:_{[0]: ValueNode<FuncGraph> 1282__tuple_getitem_by_number_2683, [1]: CNode_2681, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.880.063 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1283_✓↓get_query_key_value_tensors_2882:_{[0]: ValueNode<FuncGraph> 1287__tuple_getitem_by_number_2878, [1]: CNode_2875, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.880.289 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@596_dsa_forward_3943:_{[0]: ValueNode<FuncGraph> 883__tuple_getitem_by_number_3909, [1]: CNode_3908, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.880.571 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1022__topk_with_bias_2370:_{[0]: ValueNode<FuncGraph> 1025__tuple_getitem_by_number_2364, [1]: CNode_2360, [2]: ValueNode<Int64Imm> 0}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.881.153 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@641_get_qk_index_3657:_{[0]: ValueNode<FuncGraph> 730__tuple_getitem_by_number_3487, [1]: CNode_3486, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.881.176 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@641_get_qk_index_3657:_{[0]: ValueNode<FuncGraph> 731__tuple_getitem_by_number_3505, [1]: CNode_3504, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.881.195 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@641_get_qk_index_3657:_{[0]: ValueNode<FuncGraph> 732__tuple_getitem_by_number_3523, [1]: CNode_3522, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.881.646 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1259_✓qkv_up_proj_and_rope_apply_2830:_{[0]: ValueNode<FuncGraph> 1268__tuple_getitem_by_number_2826, [1]: CNode_2824, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.881.810 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1054_3↓dsa_forward_3031:_{[0]: ValueNode<FuncGraph> 1071__tuple_getitem_by_number_3003, [1]: CNode_3002, [2]: ValueNode<Int64Imm> 0}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.881.833 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1054_3↓dsa_forward_3031:_{[0]: ValueNode<FuncGraph> 1072__tuple_getitem_by_number_3004, [1]: CNode_3002, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.881.853 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@1054_3↓dsa_forward_3031:_{[0]: ValueNode<FuncGraph> 1073__tuple_getitem_by_number_3005, [1]: CNode_3002, [2]: ValueNode<Int64Imm> 2}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.883.553 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@781_2↓get_query_key_value_tensors_3853:_{[0]: ValueNode<FuncGraph> 862__tuple_getitem_by_number_3684, [1]: CNode_3682, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.883.643 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@863_✓↓get_query_key_value_tensors_3883:_{[0]: ValueNode<FuncGraph> 867__tuple_getitem_by_number_3879, [1]: CNode_3876, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.884.016 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@570__topk_with_bias_3365:_{[0]: ValueNode<FuncGraph> 573__tuple_getitem_by_number_3359, [1]: CNode_3355, [2]: ValueNode<Int64Imm> 0}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.884.668 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1613] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@839_✓qkv_up_proj_and_rope_apply_3831:_{[0]: ValueNode<FuncGraph> 848__tuple_getitem_by_number_3827, [1]: CNode_3825, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.884.829 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@602_3↓dsa_forward_4032:_{[0]: ValueNode<FuncGraph> 619__tuple_getitem_by_number_4004, [1]: CNode_4003, [2]: ValueNode<Int64Imm> 0}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.884.851 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@602_3↓dsa_forward_4032:_{[0]: ValueNode<FuncGraph> 620__tuple_getitem_by_number_4005, [1]: CNode_4003, [2]: ValueNode<Int64Imm> 1}
[WARNING] ANALYZER(56203,ffffa4786be0,python):2026-07-14-06:47:29.884.871 [mindspore/ccsrc/frontend/jit/ps/static_analysis/auto_monad.cc:1606] ClearIsolatedNodes] Some side effect nodes were eliminated by mistake. The node is:@602_3↓dsa_forward_4032:_{[0]: ValueNode<FuncGraph> 621__tuple_getitem_by_number_4006, [1]: CNode_4003, [2]: ValueNode<Int64Imm> 2}
[WARNING] PARSER(56203,ffffa4786be0,python):2026-07-14-06:47:42.296.973 [mindspore/ccsrc/frontend/jit/ps/parse/data_converter.cc:661] CheckAPI] The mint interface reshape was called, and the operators under this interface have different view capabilities on pynative and graph mode. Use this interface with caution in graph mode, as it may produce unexpected results. For more information, please refer to: https://www.mindspore.cn/docs/en/master/features/view.html

[WARNING] DEVICE(56203,ffffa4786be0,python):2026-07-14-06:49:02.340.398 [mindspore/ccsrc/plugin/ascend/res_manager/mem_manager/abstract_ascend_memory_pool_support.cc:48] SetMemPoolBlockSize] Memory pool block size 59055800320 is bigger than currently available maximum memory 57982058496, and the actual effective value will be 57982058496
GD===========>[cmhc_bprop] called, dhin/dhPost/dhRes:
Tensor(shape=[3], dtype=Int64, value=[4096    1 3584])
Tensor(shape=[3], dtype=Int64, value=[4096    1    4])
Tensor(shape=[3], dtype=Int64, value=[4096    1   16])
GD===========>[cmhc_bprop] hPre/hcBeforeNorm/invRms:
Tensor(shape=[3], dtype=Int64, value=[4096    1    4])
Tensor(shape=[3], dtype=Int64, value=[4096    1   32])
Tensor(shape=[3], dtype=Int64, value=[4096    1    1])
GD===========>[cmhc_bprop] grad_op done, gradX/gradPhi/gradAlpha/gradBias:
Tensor(shape=[4], dtype=Int64, value=[4096    1    4 3584])
Tensor(shape=[2], dtype=Int64, value=[   32 14336])
Tensor(shape=[1], dtype=Int64, value=[3])
Tensor(shape=[1], dtype=Int64, value=[32])
[ERROR] DEVICE(56203,fffdcb7ef1a0,python):2026-07-14-06:49:24.224.941 [mindspore/ccsrc/plugin/ascend/res_manager/hal_manager/ascend_err_manager.cc:147] TaskExceptionCallback] Run Task failed, task_id: 1255, stream_id: 3, tid: 56203, device_id: 8, retcode: 507015 (aicore exception)
[ERROR] DEVICE(56203,ffff2604f1a0,python):2026-07-14-06:49:24.225.808 [mindspore/ccsrc/plugin/ascend/res_manager/stream_manager/ascend_stream_manager.cc:301] SyncStream] Has set launch blocking, but synchronous stream still failed. Please save the complete log information to further identify the specific error cause.
[ERROR] RUNTIME_FRAMEWORK(56203,ffff2604f1a0,python):2026-07-14-06:49:24.226.279 [mindspore/ccsrc/backend/ms_backend/runtime/actors/base/kernel_async_launch_actor.cc:59] LaunchKernelV2] Failed to launch kernel: Gradients/recompute_Default/network-MFTrainOneStepCell/network-DataOrderWrapperCell/network-TrainingTeleChat4MoeForCausalLM/model-GPTModel/decoder-TransformerBlock/layers-CellList/3-TransformerLayer/attn_hc-HyperConnectionModuleAscendFusedCmhc/mhc_pre_cmhc_op-ManifoldConstrainedHyperConnectionPreCmhc/Grad_Custom/Custom-op0 and catch exception: Sync run failed, detail: EZ9999: Inner Error!
EZ9999[PID: 56203] 2026-07-14-06:49:21.811.059 (EZ9999):  The error from device(chipId:4, dieId:0), serial number is 54, there is an exception of fftsplus aicore error, core id is 0, error code = 0x800000, dump info: pc start: 0x12e082af4000, current: 0x12e082af4184, vec error info: 0, mte error info: 0xd006000032, ifu error info: 0x650f8da8379c0, ccu error info: 0xc37061c04801d02, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12e100000080.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:645]
        TraceBack (most recent call last):
       The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x6000032, fixp_error1 info: 0xd0, fsmId:0, tslot:7, thread:0, ctxid:0, blk:14, sublk:0, subErrType:4.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:658]
       The error from device(chipId:4, dieId:0), serial number is 55, there is an exception of fftsplus aivector error, core id is 1, error code = 0x800000, dump info: pc start: 0x12e082af9c50, current: 0x12e082afba54, vec error info: 0x4a00000088, mte error info: 0xee06000093, ifu error info: 0x5ae000f00ebc0, ccu error info: 0x580e802577000093, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12e100000080.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:645]
       The extend info: errcode:(0x800000, 0x4000, 0) errorStr: The DDR address of the MTE instruction is out of range.CCU instruction address check error. fixp_error0 info: 0x6000093, fixp_error1 info: 0xee, fsmId:0, tslot:7, thread:0, ctxid:0, blk:19, sublk:0, subErrType:4.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:658]
       Kernel task happen error, retCode=0x26, [aicore exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1728]
       rtStreamSynchronizeWithTimeout execution failed, reason=aicore exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
       synchronize stream with timeout failed, runtime result = 507015[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]

----------------------------------------------------
- The Function Call Stack: (For framework developers)
----------------------------------------------------
Corresponding forward node candidate:
 - In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/ops/mhc_pre_cmhc.py:247~250, 15~9/        return self._custom_op(/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/ops/mhc_pre_cmhc.py:244~250, 4~9/    def construct(self, x, phi, alpha, bias, perm_mats,/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/transformer/hyper_connection.py:568, 34~54/        aggregated, hPost, hRes = self.mhc_pre_cmhc_op(/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/transformer/hyper_connection.py:535~579, 4~40/    def forward_func(self, hidden_states, mapping_weight,/

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/ascend/kernel_executor/ascend_kernel_executor.cc:1336 LaunchKernel
#dmsg#The Function Call Stack:#dmsg#Corresponding forward node candidate:
 - In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/ops/mhc_pre_cmhc.py:247~250, 15~9/        return self._custom_op(/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/ops/mhc_pre_cmhc.py:244~250, 4~9/    def construct(self, x, phi, alpha, bias, perm_mats,/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/transformer/hyper_connection.py:568, 34~54/        aggregated, hPost, hRes = self.mhc_pre_cmhc_op(/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/transformer/hyper_connection.py:535~579, 4~40/    def forward_func(self, hidden_states, mapping_weight,/

Traceback (most recent call last):
  File "/lgsl_data/dianxin_telechat/mindformers/run_mindformer.py", line 356, in <module>
    main(config_)
  File "/lgsl_data/dianxin_telechat/mindformers/run_mindformer.py", line 80, in main
    trainer.train()
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/_checkparam.py", line 1397, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/lgsl_data/dianxin_telechat/mindformers/mindformers/trainer/trainer.py", line 500, in train
    self.trainer.train(
  File "/lgsl_data/dianxin_telechat/mindformers/mindformers/trainer/causal_language_modeling/causal_language_modeling.py", line 108, in train
    self.training_process(
  File "/lgsl_data/dianxin_telechat/mindformers/mindformers/trainer/base_trainer.py", line 1687, in training_process
    model.train(config.runner_config.epochs, dataset,
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/train/model.py", line 1369, in train
    self._train(epoch,
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/train/model.py", line 324, in wrapper
    return func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/train/model.py", line 123, in wrapper
    func(self, *args, **kwargs)
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/train/model.py", line 1071, in _train
    self._train_dataset_sink_process(epoch, train_dataset, list_callback,
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/train/model.py", line 1144, in _train_dataset_sink_process
    outputs = train_network(*inputs)
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1371, in __call__
    out = self.compile_and_run(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1767, in compile_and_run
    return _cell_graph_executor(self, *new_args, phase=self.phase)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/graph/api.py", line 2426, in __call__
    return self.run(obj, *args, phase=phase)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/graph/api.py", line 2477, in run
    return self._exec_pip(obj, *args, phase=phase_real)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/graph/api.py", line 197, in wrapper
    results = fn(*arg, **kwargs)
              ^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/mindspore/graph/api.py", line 2457, in _exec_pip
    return self._graph_executor(args, phase)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Sync run failed, detail: EZ9999: Inner Error!
EZ9999[PID: 56203] 2026-07-14-06:49:21.811.059 (EZ9999):  The error from device(chipId:4, dieId:0), serial number is 54, there is an exception of fftsplus aicore error, core id is 0, error code = 0x800000, dump info: pc start: 0x12e082af4000, current: 0x12e082af4184, vec error info: 0, mte error info: 0xd006000032, ifu error info: 0x650f8da8379c0, ccu error info: 0xc37061c04801d02, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12e100000080.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:645]
        TraceBack (most recent call last):
       The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x6000032, fixp_error1 info: 0xd0, fsmId:0, tslot:7, thread:0, ctxid:0, blk:14, sublk:0, subErrType:4.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:658]
       The error from device(chipId:4, dieId:0), serial number is 55, there is an exception of fftsplus aivector error, core id is 1, error code = 0x800000, dump info: pc start: 0x12e082af9c50, current: 0x12e082afba54, vec error info: 0x4a00000088, mte error info: 0xee06000093, ifu error info: 0x5ae000f00ebc0, ccu error info: 0x580e802577000093, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12e100000080.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:645]
       The extend info: errcode:(0x800000, 0x4000, 0) errorStr: The DDR address of the MTE instruction is out of range.CCU instruction address check error. fixp_error0 info: 0x6000093, fixp_error1 info: 0xee, fsmId:0, tslot:7, thread:0, ctxid:0, blk:19, sublk:0, subErrType:4.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:658]
       Kernel task happen error, retCode=0x26, [aicore exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1728]
       rtStreamSynchronizeWithTimeout execution failed, reason=aicore exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
       synchronize stream with timeout failed, runtime result = 507015[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]

----------------------------------------------------
- The Function Call Stack: (For framework developers)
----------------------------------------------------
Corresponding forward node candidate:
 - In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/ops/mhc_pre_cmhc.py:247~250, 15~9/        return self._custom_op(/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/ops/mhc_pre_cmhc.py:244~250, 4~9/    def construct(self, x, phi, alpha, bias, perm_mats,/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/transformer/hyper_connection.py:568, 34~54/        aggregated, hPost, hRes = self.mhc_pre_cmhc_op(/
   In file /lgsl_data/dianxin_telechat/mindformers/mindformers/parallel_core/training_graph/transformer/hyper_connection.py:535~579, 4~40/    def forward_func(self, hidden_states, mapping_weight,/

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/ascend/kernel_executor/ascend_kernel_executor.cc:1336 LaunchKernel

[ERROR] DEVICE(56203,ffffa4786be0,python):2026-07-14-06:49:27.494.030 [mindspore/ccsrc/plugin/ascend/res_manager/stream_manager/ascend_stream_manager.cc:345] SyncAllStreams] Has set launch blocking, but synchronous stream still failed. Please save the complete log information to further identify the specific error cause.
[ERROR] ME(56203,ffffa4786be0,python):2026-07-14-06:49:27.494.066 [mindspore/ccsrc/runtime/hardware_abstract/device_context/device_context_manager.cc:527] WaitTaskFinishOnDevice] SyncStream failed
[root@YGA-A2-4FZH06-1411-0512-HW-AL900A3-GPU-ARM-141-CJD mindformers]#
