�	\�z�6@\�z�6@!\�z�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-\�z�6@��	/��3@1K�|%��?A)�7Ӆ�?ILP÷�n @*	33333�M@2U
Iterator::Model::ParallelMapV2��ׁsF�?!������@@)��ׁsF�?1������@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�q����?!��y;C:@)g��j+��?1��&�l�3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!�|ז�4@)�j+��݃?1��c�xT0@:Preprocessing2F
Iterator::Model_�Qڛ?!�?ƊG�F@)���_vO~?1yTn�s�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!��y;C@)�q����o?1��y;C@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?�ܵ�|�?!7�9u�K@)F%u�k?1��8��8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�Ω��@){�G�zd?1�Ω��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o_��?!��pN<@)/n��R?1�Kh/��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��%<��X@Q/�1���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��	/��3@��	/��3@!��	/��3@      ��!       "	K�|%��?K�|%��?!K�|%��?*      ��!       2	)�7Ӆ�?)�7Ӆ�?!)�7Ӆ�?:	LP÷�n @LP÷�n @!LP÷�n @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��%<��X@y/�1���?�"5
sequential/dense/MatMulMatMul�N���L�?!�N���L�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�Sm'L�?!̭��vL�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�F\�dƘ?!����~�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�F\�dƘ?!����W�?"7
sequential/dense_1/MatMulMatMul�F\�dƘ?!�q`�p�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrado=(�鰗?!.y�Y�f�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamZ�I(�ѓ?!y�����?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGradZ�I(�ѓ?!����8[�?"E
'gradient_tape/sequential/dense_1/MatMulMatMullX��Z�?!��b����?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�ĳyY��?!(7Q�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�o�ҠW@y�X�����?"�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 