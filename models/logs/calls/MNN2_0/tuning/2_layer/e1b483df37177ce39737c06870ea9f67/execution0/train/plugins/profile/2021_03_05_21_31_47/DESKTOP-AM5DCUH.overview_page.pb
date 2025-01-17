�	G���)7@G���)7@!G���)7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-G���)7@�H��ڲ4@1�J[\�3�?AM�St$�?Ihur��@*	gffff&J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��_�L�?!�E����C@)�N@aÓ?1W�l��sB@:Preprocessing2U
Iterator::Model::ParallelMapV2�HP��?!O ���S7@)�HP��?1O ���S7@:Preprocessing2F
Iterator::Model���&�?!�\L��A@)9��v��z?1��{�I�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�J�4�?!��٩0@)9��v��z?1��{�I�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J�4�?!��٩P@)-C��6j?1=iXdy@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!���('@)ŏ1w-!_?1���('@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I+��?!p��;RE@)a2U0*�S?1�B[@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!=iXdy�?)-C��6J?1=iXdy�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��7j�?)Ǻ���F?1��7j�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��+h�X@Q�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�H��ڲ4@�H��ڲ4@!�H��ڲ4@      ��!       "	�J[\�3�?�J[\�3�?!�J[\�3�?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	hur��@hur��@!hur��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��+h�X@y�����?�"5
sequential/dense/MatMulMatMul�u�3�#�?!�u�3�#�?0"C
%gradient_tape/sequential/dense/MatMulMatMul8��-p�?!�}�0��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul8��-p�?!(@�G_��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul8��-p�?!b�N����?"7
sequential/dense_1/MatMulMatMul8��-p�?!�bƺg��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�G�D(��?!�K`���?"7
sequential/dense_2/MatMulMatMul�fW9LD�?!{8�J���?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam8��-p�?!")GP��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch8��-p�?!�Vr��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad8��-p�?!8��-p�?Q      Y@Y�C=�C=0@a��
��T@q�}���]X@y;�����?"�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 