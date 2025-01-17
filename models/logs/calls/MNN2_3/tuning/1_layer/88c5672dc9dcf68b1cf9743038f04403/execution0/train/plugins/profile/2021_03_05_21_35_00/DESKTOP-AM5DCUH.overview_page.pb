�	ٖg)56@ٖg)56@!ٖg)56@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ٖg)56@9% &��3@1��-u���?AbX9�Ȧ?I"R�.�� @*	������G@2U
Iterator::Model::ParallelMapV2�?�߾�?!ػ
ܞ�<@)�?�߾�?1ػ
ܞ�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!��;E;@)A��ǘ��?1q��;E7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!��Gn6@)F%u�{?1����F�+@:Preprocessing2F
Iterator::Model�0�*��?!/=��E@)-C��6z?1}���*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!<Ь�0!@)	�^)�p?1<Ь�0!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!��J�L@)�����g?1S)ϖ�Q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��sHM0@)��H�}]?1��sHM0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!H7���8@)/n��R?1�e*��r@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIL��8��X@Q�ٓ�c�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	9% &��3@9% &��3@!9% &��3@      ��!       "	��-u���?��-u���?!��-u���?*      ��!       2	bX9�Ȧ?bX9�Ȧ?!bX9�Ȧ?:	"R�.�� @"R�.�� @!"R�.�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qL��8��X@y�ٓ�c�?�"5
sequential/dense/MatMulMatMul1]��?!1]��?0"C
%gradient_tape/sequential/dense/MatMulMatMul'�elƤ?!��c�9_�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch1]��?!��)�;ݽ?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul1]��?!�'�؞-�?"7
sequential/dense_1/MatMulMatMul1]��?!HS�ȟl�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad?t̸ǔ?!)�i��?")
sequential/CastCast��f;,�?!��ʘZ��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam:ք��)�?!LAS����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam:ք��)�?!������?"!
Adam/PowPow:ք��)�?!�c)��?Q      Y@Y��/Ċ�0@a�	�N]�T@qb�CTX@y>�o�;�?"�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 