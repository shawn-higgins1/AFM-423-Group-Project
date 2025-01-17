�	5|�27@5|�27@!5|�27@	����ը�?����ը�?!����ը�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails65|�27@�=�Х4@1�D�A��?A_)�Ǻ�?Iz�(�@Y�W�\t?*	53333sN@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!�̹��A@)/n���?1'A��[�<@:Preprocessing2U
Iterator::Model::ParallelMapV2�!��u��?!�L��'7@)�!��u��?1�L��'7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�!��u��?!�L��'7@)tF��_�?1Ȥx�L�3@:Preprocessing2F
Iterator::ModelDio��ɔ?!������@@)a��+ey?1�ؾz\$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!a���@)�J�4q?1a���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDio��ɤ?!������P@)���_vOn?1��;�XM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!'A��[�@)/n��b?1'A��[�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+e�X�?!�8D�B@)����MbP?1����E�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����ը�?I���Z��X@Q�)��D�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�=�Х4@�=�Х4@!�=�Х4@      ��!       "	�D�A��?�D�A��?!�D�A��?*      ��!       2	_)�Ǻ�?_)�Ǻ�?!_)�Ǻ�?:	z�(�@z�(�@!z�(�@B      ��!       J	�W�\t?�W�\t?!�W�\t?R      ��!       Z	�W�\t?�W�\t?!�W�\t?b      ��!       JGPUY����ը�?b q���Z��X@y�)��D�?�"5
sequential/dense/MatMulMatMulįD���?!įD���?0"C
%gradient_tape/sequential/dense/MatMulMatMul���Yg��?!:J
��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���Yg��?!�<�UE��?0"7
sequential/dense_1/MatMulMatMul���Yg��?!um�<~�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�h����?!�@��ߪ�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�z��5�?!p^�Q�?"7
sequential/dense_2/MatMulMatMul��C04�?!��f)��?0"!
Adam/PowPow���Yg��?!h���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���Yg��?!�����5�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���Yg��?!�p�u4��?Q      Y@Y>����/@aX�i��U@q�k��W@yQ�T|��?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 