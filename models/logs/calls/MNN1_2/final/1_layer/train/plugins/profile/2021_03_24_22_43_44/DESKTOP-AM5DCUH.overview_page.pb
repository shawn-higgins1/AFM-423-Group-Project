�	" 8��!5@" 8��!5@!" 8��!5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-" 8��!5@�3��k�2@1���N@�?A���9̧?I���P��@*	gffff�J@2U
Iterator::Model::ParallelMapV2�J�4�?!���?@)�J�4�?1���?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!ʯ�rT9@)M�St$�?1o2�ad35@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea��+e�?!�b�C7@)��ǘ���?1�l&��d.@:Preprocessing2F
Iterator::Model���Mb�?!�����F@)S�!�uq{?1�o{�$)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!�u��" @)"��u��q?1�u��" @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2�%䃞?!. 	y�K@)Ǻ���f?1��\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!o�=C��@)/n��b?1o�=C��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���S㥋?!ʯ�rT9@)/n��R?1o�=C�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�Yׂ�X@Q`�)J��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�3��k�2@�3��k�2@!�3��k�2@      ��!       "	���N@�?���N@�?!���N@�?*      ��!       2	���9̧?���9̧?!���9̧?:	���P��@���P��@!���P��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�Yׂ�X@y`�)J��?�"5
sequential/dense/MatMulMatMulF?��N�?!F?��N�?0"C
%gradient_tape/sequential/dense/MatMulMatMulwvT�1ԣ?!޽ɟt�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�_��n�?!�3m�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�)�=ɘ?!m=�=�O�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamwvT�1ԓ?!<̐nG��?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCastwvT�1ԓ?![{��D�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradwvT�1ԓ?!��e�S��?"E
'gradient_tape/sequential/dense_1/MatMulMatMulwvT�1ԓ?!�xP�9�?0"7
sequential/dense_1/MatMulMatMulwvT�1ԓ?!x;2`��?0"&
	truediv_1RealDivA�/����?!<Nl��?Q      Y@Y��/Ċ�0@a�	�N]�T@qtz��lX@y�sc���?"�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 