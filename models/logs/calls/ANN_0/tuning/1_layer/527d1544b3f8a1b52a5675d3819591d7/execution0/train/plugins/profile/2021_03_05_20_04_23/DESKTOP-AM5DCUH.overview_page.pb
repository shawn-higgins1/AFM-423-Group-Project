�	�磌�(3@�磌�(3@!�磌�(3@	���Z�R�?���Z�R�?!���Z�R�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�磌�(3@�4}v��0@1Ov3��?A��ͪ�զ?I�q6@Y!#���r?*	������P@2U
Iterator::Model::ParallelMapV2��ׁsF�?!6U��)=@)��ׁsF�?16U��)=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� �rh�?!� ��l	9@)%u��?1b�Y�D�5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr�����?!���W:@)�HP��?1�oQ���1@:Preprocessing2F
Iterator::Model�v��/�?!ϭ�U��D@)�5�;Nс?1��R<�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�I+�v?!�K�F3 @)�I+�v?1�K�F3 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���<,�?!2Rj�dM@)U���N@s?17��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�*»B@)HP�s�b?1�*»B@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���Z�R�?I{"hWZ�X@Qc	=��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�4}v��0@�4}v��0@!�4}v��0@      ��!       "	Ov3��?Ov3��?!Ov3��?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	�q6@�q6@!�q6@B      ��!       J	!#���r?!#���r?!!#���r?R      ��!       Z	!#���r?!#���r?!!#���r?b      ��!       JGPUY���Z�R�?b q{"hWZ�X@yc	=��?�"5
sequential/dense/MatMulMatMul��u=k+�?!��u=k+�?0"C
%gradient_tape/sequential/dense/MatMulMatMul%�h����?!�7�򏌻?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��4��?!�ܝ����?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�3��S�?!TC7�
�?"!
Adam/PowPow%�h���?!�X!̑H�?"7
sequential/dense_1/MatMulMatMul%�h���?!�n.aH��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�g%G�?!��A��?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCastf�Y�s�?!QcG��?"E
'gradient_tape/sequential/dense_1/MatMulMatMulf�Y�s�?!׮�k���?0"
Abs_1Abs܁N~Gr�?!������?Q      Y@Y�M�_{4@a��(�S@q�-\��/W@y�c����?"�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 