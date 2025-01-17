�	�T2 T55@�T2 T55@!�T2 T55@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�T2 T55@8��̒�2@1<0���?A,e�X�?I#LQ.��@*	������I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�0�*�?!Nz5�1�C@)���&�?1�+��B@:Preprocessing2U
Iterator::Model::ParallelMapV2�?�߾�?!S�s��:@)�?�߾�?1S�s��:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat� �	�?!W�+��-@)��0�*x?1�`���&@:Preprocessing2F
Iterator::Model+�����?!�^��B@)�����w?1`sk;�o&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?�ܵ�|�?!�3�%$O@)��_�Le?1�ȯ��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��^�@)��H�}]?1��^�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw-!�l�?!�v��.E@)��_�LU?1�ȯ��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����MbP?!��pl���?)����MbP?1��pl���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!��^��?)��H�}M?1��^��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�6�A6�X@QȽ��d�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	8��̒�2@8��̒�2@!8��̒�2@      ��!       "	<0���?<0���?!<0���?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	#LQ.��@#LQ.��@!#LQ.��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�6�A6�X@yȽ��d�?�"5
sequential/dense/MatMulMatMul�ίra
�?!�ίra
�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�M�َ(�?!g�L&x��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitcht�c��2�?!�ve�$�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradt�c��2�?!�/?�h��?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMult�c��2�?!��K	���?"7
sequential/dense_1/MatMulMatMult�c��2�?!�X[��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�M�َ(�?!�A�6'k�?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1TileO^�'�6�?!u��;���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam���F�<�?!�l��ɵ�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam���F�<�?!
�3��L�?Q      Y@Y{	�%��1@a�����T@q�q{�lX@y]g]g���?"�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 