�	��.QE5@��.QE5@!��.QE5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��.QE5@�S:X�2@1������?A��ͪ�զ?I��-sZ@*	�����YH@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate����Mb�?!�\0�Vm@@)�HP��?1�YcE$9@:Preprocessing2U
Iterator::Model::ParallelMapV246<�R�?!�5-�a6@)46<�R�?1�5-�a6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!����,�7@)n���?1��!@�4@:Preprocessing2F
Iterator::Model�� �rh�?!Mb�',tA@)�HP�x?1�YcE$)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceŏ1w-!o?!�|��$6@)ŏ1w-!o?1�|��$6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!�N��EP@)-C��6j?1;ǳƊH@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!�s�@)_�Q�[?1�s�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�� �rh�?!Mb�',tA@)����MbP?1�\0�Vm @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�0���X@Q�3�{��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�S:X�2@�S:X�2@!�S:X�2@      ��!       "	������?������?!������?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	��-sZ@��-sZ@!��-sZ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�0���X@y�3�{��?�"C
%gradient_tape/sequential/dense/MatMulMatMul*4)Q7�?!*4)Q7�?0"5
sequential/dense/MatMulMatMul*4)Q7�?!*4)Q7�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul4�s%�Q�?!w�����?"7
sequential/dense_1/MatMulMatMul4�s%�Q�?!bz���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�.Ѷ�'�?!<���`�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad*4)Q7�?!����H�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul*4)Q7�?!F����?0"$
MaximumMaximume�����?!|l )���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam?ν�R��?!`I�X��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam?ν�R��?!D&X�9��?Q      Y@Y��/Ċ�0@a�	�N]�T@q��$X@y�0h7s�?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 