�	���C�!5@���C�!5@!���C�!5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���C�!5@N�t"�2@1yͫ:��?A���JY��?I`�o`r� @*23333�H@)       =2U
Iterator::Model::ParallelMapV2�Pk�w�?!4"Y&=#<@)�Pk�w�?14"Y&=#<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�?�߾�?!��4[��;@)M�St$�?1������6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!	9?�6@)lxz�,C|?1����j�+@:Preprocessing2F
Iterator::ModelQ�|a2�?!9?���D@)_�Q�{?1|���Ň+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!���1 @)����Mbp?1���1 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipO��e�c�?!��j�oM@)�����g?1��7V{@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!�6��n@)a2U0*�c?1�6��n@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@��ǈ?!�r~8@)a2U0*�S?1�6��n@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIT[1n�X@Q��UR���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N�t"�2@N�t"�2@!N�t"�2@      ��!       "	yͫ:��?yͫ:��?!yͫ:��?*      ��!       2	���JY��?���JY��?!���JY��?:	`�o`r� @`�o`r� @!`�o`r� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qT[1n�X@y��UR���?�"5
sequential/dense/MatMulMatMulY�3�5�?!Y�3�5�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�@����?!�t[��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�ʙ?!�x�8^�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ʙ?!���\h�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchՆＴ��?!��[~Ӈ�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�@����?!R��.�?0"7
sequential/dense_1/MatMulMatMul�@����?!�+�:��?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile��9HZx�?!�'3(F?�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamv�Dl�?!@(��l.�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamv�Dl�?!d�g���?Q      Y@Y��/Ċ�0@a�	�N]�T@q+�%h�tX@yh�C�?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 