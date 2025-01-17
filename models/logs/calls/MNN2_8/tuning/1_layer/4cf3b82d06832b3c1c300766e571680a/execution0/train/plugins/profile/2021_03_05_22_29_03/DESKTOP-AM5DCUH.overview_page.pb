�	>z�}�.6@>z�}�.6@!>z�}�.6@	��
�t�?��
�t�?!��
�t�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6>z�}�.6@衶��3@1�R?o*R�?A����K�?I�Ɵ�l8@Y�� �X4m?*�����G@)       =2U
Iterator::Model::ParallelMapV29��v���?!ۍ��v#<@)9��v���?1ۍ��v#<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��_vO�?!���cj`7@)��y�):�?1h�C3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg��j+��?!wL�S9@)ŏ1w-!?1͡bAs0@:Preprocessing2F
Iterator::Model��~j�t�?!G@J��D@)�~j�t�x?1f�'�Y�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!�����!@)	�^)�p?1�����!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_�Qڛ?!���D�oM@)y�&1�l?1N6�d�M@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!͡bAs@)ŏ1w-!_?1͡bAs@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{�Pk�?!5�wL�;@)a2U0*�S?1�S{�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��
�t�?I��}d�X@Q��xI��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	衶��3@衶��3@!衶��3@      ��!       "	�R?o*R�?�R?o*R�?!�R?o*R�?*      ��!       2	����K�?����K�?!����K�?:	�Ɵ�l8@�Ɵ�l8@!�Ɵ�l8@B      ��!       J	�� �X4m?�� �X4m?!�� �X4m?R      ��!       Z	�� �X4m?�� �X4m?!�� �X4m?b      ��!       JGPUY��
�t�?b q��}d�X@y��xI��?�"5
sequential/dense/MatMulMatMulbY�?!bY�?0"C
%gradient_tape/sequential/dense/MatMulMatMul;�����?!�e����?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad&˹�d��?!��w�I�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�;8*��?!��""���?"7
sequential/dense_1/MatMulMatMul�;8*��?!.�	f���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�`��B�?!HG�����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam;�����?!o�����?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad�η���?!�B��v�?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile�y�@|�?!��^�M�?"&
	truediv_1RealDiv�y�@|�?!�|��%�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�jnߠW@yѷD}'�?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 