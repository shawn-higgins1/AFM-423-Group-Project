�	���<��9@���<��9@!���<��9@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���<��9@����6@1S�{/��?A�a��4�?I�R�Z!@*	     @Q@2U
Iterator::Model::ParallelMapV2���H�?!��,d!7@)���H�?1��,d!7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!-d!Y�8@)%u��?1��7��M5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���<,�?!(�3J��<@)tF��_�?14J��?1@:Preprocessing2F
Iterator::Model�������?!��v`�B@)�&S��?1�ځ�v`*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����?!�Mozӛ&@)�q����?1�Mozӛ&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�I+��?!}F��Q�O@)��0�*x?1Q�g��!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!��Moz�@)a2U0*�c?1��Moz�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/�$��?!�7��Mo>@)��_�LU?1�g��%�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�dJ�X@Q��M����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����6@����6@!����6@      ��!       "	S�{/��?S�{/��?!S�{/��?*      ��!       2	�a��4�?�a��4�?!�a��4�?:	�R�Z!@�R�Z!@!�R�Z!@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�dJ�X@y��M����?�"5
sequential/dense/MatMulMatMul����?!����?0"C
%gradient_tape/sequential/dense/MatMulMatMuln�?Ŭ��?!���f�̷?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�G�L�?!��5���?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�G�L�?!�#�q,�?"7
sequential/dense_1/MatMulMatMul�G�L�?!�LW�sO�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad������?!eK��gc�?"!
Adam/PowPowӍ=D�?!�yn���?")
sequential/CastCastӍ=D�?!(�*�8h�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamf���?!ͬ���?"5
sequential/dense/BiasAddBiasAddMn��h�?!�n�����?Q      Y@Y��/Ċ�0@a�	�N]�T@q�{����W@yÙ-�^�?"�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 