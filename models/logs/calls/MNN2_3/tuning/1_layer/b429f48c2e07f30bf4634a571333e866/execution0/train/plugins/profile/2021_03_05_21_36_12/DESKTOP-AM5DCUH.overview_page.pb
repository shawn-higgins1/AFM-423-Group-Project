�	2t��5@2t��5@!2t��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-2t��5@]�C��3@1�F��?A�"���?I)��q�@*dfffffG@)       =2U
Iterator::Model::ParallelMapV2S�!�uq�?!��<@)S�!�uq�?1��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!i�i�;@)�I+��?1��7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_�L�?!��8��86@)-C��6z?1��Y��Y+@:Preprocessing2F
Iterator::Model�N@aÓ?!�ꡞD@)��0�*x?1k�6k�6)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��!@)����Mbp?1��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)\���(�?!^�^aM@)�����g?1�LɔL�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�C=�C=@)ŏ1w-!_?1�C=�C=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!      9@)��_�LU?1��8��8@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���e'�X@Qt��*Ml�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	]�C��3@]�C��3@!]�C��3@      ��!       "	�F��?�F��?!�F��?*      ��!       2	�"���?�"���?!�"���?:	)��q�@)��q�@!)��q�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���e'�X@yt��*Ml�?�"5
sequential/dense/MatMulMatMulm���Ž�?!m���Ž�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��_�l�?!�Z��W��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��ϢD�?!�����?"7
sequential/dense_1/MatMulMatMulz��뢘?!9�����?0"!
Adam/PowPowDq2��6�?!b��z�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchDq2��6�?!�Z��W��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradDq2��6�?!��r�4�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�
$rP�?!*���?0"
MulMul5p`��T�?!
1Jm~�?"
Abs_1Abs�K�\R�?!���c�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�K��yX@y�Ю�˕�?"�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 