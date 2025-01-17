�	=�U�#6@=�U�#6@!=�U�#6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-=�U�#6@�l��<?3@1[�[!���?A9��m4��?Ih���@*	������E@2U
Iterator::Model::ParallelMapV2������?!�i?Y�:@)������?1�i?Y�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+e�X�?!Y���%:@)�&S��?1�]8��4@:Preprocessing2F
Iterator::Model�&S��?!�]8��D@)F%u�{?1h�bnuF.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatea2U0*��?!L`�~�6@)�HP�x?1��,�M�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicey�&1�l?!� z| @)y�&1�l?1� z| @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�
F%u�?!��w� M@)a��+ei?1��p@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�����@)HP�s�b?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!      9@)��_�LU?1��d	l�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI+ٟ��X@Q�j0q<�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�l��<?3@�l��<?3@!�l��<?3@      ��!       "	[�[!���?[�[!���?![�[!���?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	h���@h���@!h���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q+ٟ��X@y�j0q<�?�"5
sequential/dense/MatMulMatMulHƖ7���?!HƖ7���?0"C
%gradient_tape/sequential/dense/MatMulMatMul�Z�M��?!��µM�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�1R!!�?!����?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�Z�M��?!�9B�{J�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�Z�M��?!<�����?"7
sequential/dense_1/MatMulMatMul�Z�M��?!��µM�?0"$
MaximumMaximum�ò���?!�<ԋ/�?"E
(gradient_tape/mean_squared_error/truedivRealDiv�ò���?!i�Tq�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam`��Z�?!��;���?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam`��Z�?!������?Q      Y@Y��/Ċ�0@a�	�N]�T@q���0=�X@y�=�?"�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 